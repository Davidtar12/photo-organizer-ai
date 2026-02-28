import { useState, useCallback, useEffect } from 'react'
import { DuplicateSet } from './lib/types'
import { loadDuplicatesFromServer, loadMetadata, deleteFile, PaginatedResponse, bulkDeleteDuplicates } from './lib/helpers'
import { useKeyboard } from './hooks/use-keyboard'
import './App.css'

type ViewMode = 'bulk' | 'review'

export default function AppNew() {
  const [view, setView] = useState<ViewMode>('bulk')
  const [sets, setSets] = useState<DuplicateSet[]>([])
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [selected, setSelected] = useState<Set<number>>(new Set())

  const [currentIndex, setCurrentIndex] = useState(0)
  const [deleting, setDeleting] = useState(false)
  const [loadingMetadata, setLoadingMetadata] = useState(false)

  const loadMore = useCallback(async (reset = false) => {
    try {
      setLoading(true)
      const start = reset ? 0 : offset
      const resp: PaginatedResponse = await loadDuplicatesFromServer(true, start, 100)
      if (reset) {
        setSets(resp.duplicates)
        setSelected(new Set())
      } else {
        setSets(prev => [...prev, ...resp.duplicates])
      }
      setOffset(start + resp.pagination.count)
      setHasMore(resp.pagination.hasMore)
      setError(null)
    } catch (e: any) {
      setError(e.message || 'Failed to load duplicates')
    } finally {
      setLoading(false)
    }
  }, [offset])

  useEffect(() => { loadMore(true) }, [])

  useEffect(() => {
    if (view === 'review' && hasMore && !loading && currentIndex >= sets.length - 5) {
      loadMore(false)
    }
  }, [view, currentIndex, sets.length, hasMore, loading, loadMore])

  useEffect(() => {
    async function fetchMeta() {
      const s = sets[currentIndex]
      if (!s) return
      if (s.original.fileSize !== undefined) return
      try {
        setLoadingMetadata(true)
        const meta = await loadMetadata(currentIndex)
        setSets(prev => {
          const copy = [...prev]
          copy[currentIndex] = {
            ...copy[currentIndex],
            original: { ...copy[currentIndex].original, ...meta.original },
            duplicate: { ...copy[currentIndex].duplicate, ...meta.duplicate }
          }
          return copy
        })
      } finally {
        setLoadingMetadata(false)
      }
    }
    if (view === 'review') fetchMeta()
  }, [view, currentIndex, sets])

  const toggleSelect = (idx: number) => {
    setSelected(prev => {
      const s = new Set(prev)
      if (s.has(idx)) s.delete(idx)
      else s.add(idx)
      return s
    })
  }

  const selectAllVisible = () => {
    if (selected.size === sets.length) setSelected(new Set())
    else setSelected(new Set(sets.map((_, i) => i)))
  }

  const handleBulkDelete = async () => {
    if (selected.size === 0) return alert('No items selected')
    if (!confirm(`Delete ${selected.size} duplicate files? This cannot be undone!`)) return
    try {
      const pairs = Array.from(selected).map(i => ({ original: sets[i].original.path, duplicate: sets[i].duplicate.path }))
      const res = await bulkDeleteDuplicates(pairs)
      console.log(`Bulk deleted ${res.deleted_count}, errors: ${res.error_count}`)
      // Remove deleted indices
      setSets(prev => prev.filter((_, i) => !selected.has(i)))
      setSelected(new Set())
    } catch (e: any) {
      alert(`Error: ${e.message}`)
    }
  }

  const removeCurrentPair = () => {
    setSets(prev => {
      const next = prev.filter((_, i) => i !== currentIndex)
      if (currentIndex >= next.length && next.length > 0) {
        setCurrentIndex(next.length - 1)
      }
      return next
    })
  }

  const handleDeleteOne = async (type: 'original' | 'duplicate') => {
    const s = sets[currentIndex]
    if (!s) return
    const path = type === 'original' ? s.original.path : s.duplicate.path
    try {
      setDeleting(true)
      await deleteFile(path)
      removeCurrentPair()
    } catch (e: any) {
      alert(`Error: ${e.message}`)
    } finally {
      setDeleting(false)
    }
  }

  const handleDeleteBoth = async () => {
    const s = sets[currentIndex]
    if (!s) return
    if (!confirm('Delete BOTH original and duplicate? This cannot be undone!')) return
    try {
      setDeleting(true)
      await fetch('http://localhost:5000/api/delete-both', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ original: s.original.path, duplicate: s.duplicate.path })
      }).then(async r => {
        if (!r.ok) {
          const j = await r.json()
          throw new Error(j.error || 'Failed to delete both')
        }
      })
      removeCurrentPair()
    } catch (e: any) {
      alert(`Error: ${e.message}`)
    } finally {
      setDeleting(false)
    }
  }

  useKeyboard({
    onLeft: () => view === 'review' && currentIndex > 0 && setCurrentIndex(currentIndex - 1),
    onRight: () => view === 'review' && currentIndex < sets.length - 1 && setCurrentIndex(currentIndex + 1),
    onOne: () => view === 'review' && handleDeleteOne('duplicate'),
    onTwo: () => view === 'review' && handleDeleteOne('original'),
    onThree: () => view === 'review' && handleDeleteBoth(),
  })

  if (view === 'bulk') {
    return (
      <div className="bulk-view">
        <div className="bulk-header">
          <h1>📦 Bulk Duplicate Manager</h1>
          <div className="bulk-stats">Loaded: {sets.length}{hasMore ? '+' : ''}</div>
          <div className="bulk-actions">
            <button onClick={selectAllVisible}>{selected.size === sets.length ? '☑ Deselect All' : '☐ Select All'}</button>
            <button onClick={() => loadMore(false)} disabled={loading || !hasMore}>{loading ? 'Loading…' : '⬇️ Load More (100)'}</button>
              <button onClick={handleBulkDelete} disabled={selected.size === 0}>🗑️ Delete Selected</button>
          </div>
        </div>
        {error && <div className="error-message">❌ {error}</div>}
        <div className="duplicate-grid">
          {sets.map((set, i) => (
            <div key={set.id} className={`duplicate-card ${selected.has(i) ? 'selected' : ''}`}>
              <input type="checkbox" checked={selected.has(i)} onChange={() => toggleSelect(i)} />
              <div className="card-images" onClick={() => { setCurrentIndex(i); setView('review') }}>
                <div className="thumbnail">
                  <img src={`http://localhost:5000/api/image?path=${encodeURIComponent(set.original.path)}`} alt="Original" loading="lazy" />
                  <span className="label">Original</span>
                </div>
                <div className="thumbnail">
                  <img src={`http://localhost:5000/api/image?path=${encodeURIComponent(set.duplicate.path)}`} alt="Duplicate" loading="lazy" />
                  <span className="label">Duplicate</span>
                </div>
              </div>
              <div className="card-info">
                <div className="path" title={set.original.path}>{set.original.path.split('\\').pop()}</div>
              </div>
            </div>
          ))}
        </div>
        {hasMore && (
          <div className="load-more">
            <button onClick={() => loadMore(false)} disabled={loading}>{loading ? 'Loading…' : '⬇️ Load More (100)'}</button>
          </div>
        )}
      </div>
    )
  }

  const s = sets[currentIndex]
  if (!s) {
    return (
      <div style={{ padding: 24 }}>
        <button onClick={() => setView('bulk')}>{'← Back'}</button>
        <p>No more pairs.</p>
      </div>
    )
  }

  return (
    <div className="app">
      <div style={{ padding: 16, display: 'flex', gap: 8, alignItems: 'center' }}>
        <button onClick={() => setView('bulk')}>{'← Back to Bulk'}</button>
        <div style={{ color: '#666' }}>Pair {currentIndex + 1} / {sets.length}{hasMore ? '+' : ''}</div>
      </div>
      <div className="container">
        <div className="card">
          <div className="image-container">
            <img src={`http://localhost:5000/api/image?path=${encodeURIComponent(s.original.path)}`} alt="Original" className={'loaded'} />
          </div>
          <div className="details">
            <div className="meta">
              <div><strong>Path</strong><br />{s.original.path}</div>
              <div><strong>Resolution</strong><br />{s.original.width && s.original.height ? `${s.original.width}×${s.original.height}` : 'Unknown'}</div>
              <div><strong>File size</strong><br />{s.original.fileSize ? `${s.original.fileSize.toFixed(2)} MB` : 'Unknown'}</div>
            </div>
            <div className="actions">
              <button onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))} disabled={currentIndex === 0}>← Prev</button>
              <button onClick={() => handleDeleteOne('duplicate')} disabled={deleting}>✓ Keep</button>
              <button onClick={() => handleDeleteOne('original')} disabled={deleting}>⛔ Delete</button>
              <button onClick={() => setCurrentIndex(Math.min(sets.length - 1, currentIndex + 1))} disabled={currentIndex >= sets.length - 1}>Next →</button>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="image-container">
            <img src={`http://localhost:5000/api/image?path=${encodeURIComponent(s.duplicate.path)}`} alt="Duplicate" className={'loaded'} />
          </div>
          <div className="details">
            <div className="meta">
              <div><strong>Path</strong><br />{s.duplicate.path}</div>
              <div><strong>Resolution</strong><br />{s.duplicate.width && s.duplicate.height ? `${s.duplicate.width}×${s.duplicate.height}` : 'Unknown'}</div>
              <div><strong>File size</strong><br />{s.duplicate.fileSize ? `${s.duplicate.fileSize.toFixed(2)} MB` : 'Unknown'}</div>
            </div>
            <div className="actions">
              <button onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))} disabled={currentIndex === 0}>← Prev</button>
              <button onClick={() => handleDeleteOne('original')} disabled={deleting}>✓ Keep</button>
              <button onClick={() => handleDeleteOne('duplicate')} disabled={deleting}>⛔ Delete</button>
              <button onClick={() => setCurrentIndex(Math.min(sets.length - 1, currentIndex + 1))} disabled={currentIndex >= sets.length - 1}>Next →</button>
            </div>
          </div>
        </div>
      </div>
      <div style={{ textAlign: 'center', padding: 16 }}>
        <button onClick={handleDeleteBoth} disabled={deleting} style={{ background: '#dc3545', color: 'white', padding: '12px 24px', borderRadius: 8 }}>🗑️ Delete Both (3)</button>
      </div>
    </div>
  )
}
