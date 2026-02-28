import { useState, useEffect, useCallback } from 'react'
import { DuplicateSet } from '../lib/types'
import { loadDuplicatesFromServer, bulkDeleteDuplicates, PaginatedResponse } from '../lib/helpers'
import '../styles/BulkView.css'

interface BulkViewProps {
  onOpenReview: (index: number) => void
}

export function BulkView({ onOpenReview }: BulkViewProps) {
  const [duplicates, setDuplicates] = useState<DuplicateSet[]>([])
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [total, setTotal] = useState(0)

  const loadMore = useCallback(async (reset = false) => {
    try {
      setLoading(true)
      const newOffset = reset ? 0 : offset
      const response: PaginatedResponse = await loadDuplicatesFromServer(true, newOffset, 100)
      
      if (reset) {
        setDuplicates(response.duplicates)
        setSelected(new Set())
      } else {
        setDuplicates([...duplicates, ...response.duplicates])
      }
      
      setOffset(newOffset + response.pagination.count)
      setHasMore(response.pagination.hasMore)
      setTotal(response.pagination.total)
      setError(null)
    } catch (err: any) {
      setError(err.message || 'Failed to load duplicates')
    } finally {
      setLoading(false)
    }
  }, [offset, duplicates])

  useEffect(() => {
    loadMore(true)
  }, [])

  const toggleSelect = (index: number) => {
    const newSelected = new Set(selected)
    if (newSelected.has(index)) {
      newSelected.delete(index)
    } else {
      newSelected.add(index)
    }
    setSelected(newSelected)
  }

  const toggleSelectAll = () => {
    if (selected.size === duplicates.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(duplicates.map((_, i) => i)))
    }
  }

  const handleBulkDelete = async () => {
    if (selected.size === 0) {
      alert('No items selected')
      return
    }

    if (!confirm(`Delete ${selected.size} duplicate files? This cannot be undone!`)) {
      return
    }

    setDeleting(true)
    try {
      const pairs = Array.from(selected).map(index => ({
        original: duplicates[index].original.path,
        duplicate: duplicates[index].duplicate.path
      }))

      const result = await bulkDeleteDuplicates(pairs)
      
      alert(`✓ Deleted ${result.deleted_count} files${result.error_count > 0 ? `\n⚠ ${result.error_count} errors` : ''}`)
      
      // Reload fresh data
      await loadMore(true)
    } catch (err: any) {
      alert(`✗ Error: ${err.message}`)
    } finally {
      setDeleting(false)
    }
  }

  if (error) {
    return (
      <div className="bulk-view">
        <div className="error-message">❌ {error}</div>
        <button onClick={() => loadMore(true)}>Retry</button>
      </div>
    )
  }

  return (
    <div className="bulk-view">
      <div className="bulk-header">
        <h1>📦 Bulk Duplicate Manager</h1>
        <div className="bulk-stats">
          Loaded: {duplicates.length} / {total} pairs
        </div>
        <div className="bulk-actions">
          <button onClick={toggleSelectAll} disabled={deleting}>
            {selected.size === duplicates.length ? '☑ Deselect All' : '☐ Select All'}
          </button>
          <button 
            onClick={handleBulkDelete} 
            disabled={selected.size === 0 || deleting}
            className="delete-btn"
          >
            {deleting ? '🗑️ Deleting...' : `🗑️ Delete ${selected.size} Duplicates`}
          </button>
        </div>
      </div>

      {loading && duplicates.length === 0 ? (
        <div className="loading-spinner">Loading duplicates...</div>
      ) : (
        <>
          <div className="duplicate-grid">
            {duplicates.map((set, index) => (
              <div 
                key={set.id} 
                className={`duplicate-card ${selected.has(index) ? 'selected' : ''}`}
              >
                <input
                  type="checkbox"
                  checked={selected.has(index)}
                  onChange={() => toggleSelect(index)}
                  onClick={(e) => e.stopPropagation()}
                />
                <div className="card-images" onClick={() => onOpenReview(index)}>
                  <div className="thumbnail">
                    <img
                      src={`http://localhost:5000/api/image?path=${encodeURIComponent(set.original.path)}`}
                      alt="Original"
                      loading="lazy"
                    />
                    <span className="label">Original</span>
                  </div>
                  <div className="thumbnail">
                    <img
                      src={`http://localhost:5000/api/image?path=${encodeURIComponent(set.duplicate.path)}`}
                      alt="Duplicate"
                      loading="lazy"
                    />
                    <span className="label">Duplicate</span>
                  </div>
                </div>
                <div className="card-info">
                  <div className="path" title={set.original.path}>
                    {set.original.path.split('\\').pop()}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {hasMore && (
            <div className="load-more">
              <button onClick={() => loadMore(false)} disabled={loading}>
                {loading ? 'Loading...' : '⬇️ Load More (100)'}
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
