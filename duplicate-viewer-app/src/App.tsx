/* BEGIN-OLD-BLOCK (temporarily commented to remove duplicate module)
import { useState, useCallback, useEffect } from 'react'
import { DuplicateSet } from './lib/types'
import { loadDuplicatesFromServer, loadMetadata, deleteFile, exportDecisions, PaginatedResponse } from './lib/helpers'
import { useKeyboard } from './hooks/use-keyboard'
import './App.css'

type ViewMode = 'bulk' | 'review'

export default function App() {
  // Global state
  const [view, setView] = useState<ViewMode>('bulk')
  const [sets, setSets] = useState<DuplicateSet[]>([])
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Bulk view state
  const [selected, setSelected] = useState<Set<number>>(new Set())

  // Review view state
  const [currentIndex, setCurrentIndex] = useState(0)
  const [deleting, setDeleting] = useState(false)
  const [loadingMetadata, setLoadingMetadata] = useState(false)

  // Load a page of 100 with filtering (skip deleted)
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

  // Initial load
  useEffect(() => {
    loadMore(true)
  }, [])

  // When near end during review, prefetch next page
  useEffect(() => {
    if (view === 'review' && hasMore && !loading && currentIndex >= sets.length - 5) {
      loadMore(false)
    }
  }, [view, currentIndex, sets.length, hasMore, loading, loadMore])

  // Lazy metadata load for current pair in review
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
      } catch (e) {
        // non-fatal
        console.warn('Metadata load failed', e)
      } finally {
        setLoadingMetadata(false)
      }
    }
    if (view === 'review') fetchMeta()
  }, [view, currentIndex, sets])

  // Bulk helpers
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

  // Review actions
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
      const res = await deleteFile(path)
      console.log(res.message)
      // Immediately remove pair from list to avoid blinking
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
      // Call existing endpoint directly
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

  // Keyboard shortcuts in review mode
  useKeyboard({
    onLeft: () => view === 'review' && currentIndex > 0 && setCurrentIndex(currentIndex - 1),
    onRight: () => view === 'review' && currentIndex < sets.length - 1 && setCurrentIndex(currentIndex + 1),
    onOne: () => view === 'review' && handleDeleteOne('duplicate'), // keep original -> delete duplicate
    onTwo: () => view === 'review' && handleDeleteOne('original'),  // keep duplicate -> delete original
    onThree: () => view === 'review' && handleDeleteBoth(),
  })

  // Render bulk view
  if (view === 'bulk') {
    return (
      <div className="bulk-view">
        <div className="bulk-header">
          <h1>📦 Bulk Duplicate Manager</h1>
          <div className="bulk-stats">Loaded: {sets.length}{hasMore ? '+' : ''}</div>
          <div className="bulk-actions">
            <button onClick={selectAllVisible}>{selected.size === sets.length ? '☑ Deselect All' : '☐ Select All'}</button>
            <button onClick={() => loadMore(false)} disabled={loading || !hasMore}>{loading ? 'Loading…' : '⬇️ Load More (100)'}</button>
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

  // Render review view
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
        {/* Original card */}
        <div className={`card`}>
          <div className="image-container">
            <img
              src={`http://localhost:5000/api/image?path=${encodeURIComponent(s.original.path)}`}
              alt="Original"
              className={'loaded'}
            />
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

        {/* Duplicate card */}
        <div className={`card`}>
          <div className="image-container">
            <img
              src={`http://localhost:5000/api/image?path=${encodeURIComponent(s.duplicate.path)}`}
              alt="Duplicate"
              className={'loaded'}
            />
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

import { useState, useCallback, useEffect } from 'react'
import { DuplicateSet, ExportData } from './lib/types'
import { loadDuplicatesFromServer, loadMetadata, deleteFile, exportDecisions, PaginatedResponse } from './lib/helpers'
import { useKeyboard } from './hooks/use-keyboard'
*/
import { BulkView } from './components/BulkView'
import './App.css'

type ViewMode = 'bulk' | 'review'

function App() {
  const [viewMode, setViewMode] = useState<ViewMode>('bulk')
  const [duplicateSets, setDuplicateSets] = useState<DuplicateSet[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [loadingMetadata, setLoadingMetadata] = useState(false)
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(false)

  const sets = duplicateSets
  const currentSet = sets[currentIndex]

  const reviewedCount = sets.filter(
    (set) => set.original.decision !== 'undecided' || set.duplicate.decision !== 'undecided'
  ).length

  // Load more duplicates (pagination)
  const loadMore = useCallback(async (reset = false) => {
    try {
      setLoading(true)
      const newOffset = reset ? 0 : offset
      const response: PaginatedResponse = await loadDuplicatesFromServer(true, newOffset, 100)
      
      if (reset) {
        setDuplicateSets(response.duplicates)
      } else {
        setDuplicateSets([...duplicateSets, ...response.duplicates])
      }
      
      setOffset(newOffset + response.pagination.count)
      setHasMore(response.pagination.hasMore)
      setError(null)
    } catch (err: any) {
      setError(err.message || 'Failed to load duplicates')
    } finally {
      setLoading(false)
    }
  }, [offset, duplicateSets])

  // Load duplicates list from server on mount
  useEffect(() => {
    loadMore(true)
  }, [])

  // Auto-load more when approaching end
  useEffect(() => {
    if (viewMode === 'review' && currentIndex >= sets.length - 5 && hasMore && !loading) {
      loadMore(false)
    }
  }, [currentIndex, sets.length, hasMore, loading, viewMode])

  if (viewMode === 'bulk') {
    return <BulkView onSwitchToReview={() => setViewMode('review')} />
  }

  // Reload duplicates list from server (WITH filtering to skip deleted files)
  const reloadDuplicates = useCallback(async () => {
    await loadMore(true)
  }, [])
        setDuplicateSets(data)
        setError(null)
      } catch (err: any) {
        setError(err.message || 'Failed to load duplicates')
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  // Lazy load metadata for current set (only when viewing)
  useEffect(() => {
    async function loadCurrentMetadata() {
      if (!currentSet || currentSet.original.fileSize !== undefined) {
        return // Already loaded
      }

      try {
        setLoadingMetadata(true)
        const metadata = await loadMetadata(currentIndex)
        
        const newSets = [...sets]
        newSets[currentIndex] = {
          ...newSets[currentIndex],
          original: { ...newSets[currentIndex].original, ...metadata.original },
          duplicate: { ...newSets[currentIndex].duplicate, ...metadata.duplicate }
        }
        setDuplicateSets(newSets)
      } catch (err: any) {
        console.error('Failed to load metadata:', err)
      } finally {
        setLoadingMetadata(false)
      }
    }

    loadCurrentMetadata()
  }, [currentIndex, currentSet, sets])

  const handleUpdateDecision = useCallback(
    async (imageType: 'original' | 'duplicate', decision: 'keep' | 'delete') => {
      const newSets = [...sets]
      const set = newSets[currentIndex]
      
      if (imageType === 'original') {
        set.original.decision = decision
        if (decision === 'keep') {
          set.duplicate.decision = 'delete'
        }
      } else {
        set.duplicate.decision = decision
        if (decision === 'keep') {
          set.original.decision = 'delete'
        }
      }
      
      setDuplicateSets(newSets)

      // If decision is delete, immediately delete the file
      if (decision === 'delete') {
        const pathToDelete = imageType === 'original' ? set.original.path : set.duplicate.path
        try {
          setDeleting(true)
          const result = await deleteFile(pathToDelete)
          console.log(result.message)
          alert(`✓ ${result.message}`)
          
          // Update exists status
          if (imageType === 'original') {
            set.original.exists = false
          } else {
            set.duplicate.exists = false
          }
          setDuplicateSets([...newSets])
          
          // Remove this pair from the list (one file deleted, can't compare anymore)
          const updatedSets = newSets.filter((_, idx) => idx !== currentIndex)
          setDuplicateSets(updatedSets)
          
          // Adjust index if needed
          if (currentIndex >= updatedSets.length && updatedSets.length > 0) {
            setCurrentIndex(updatedSets.length - 1)
          }
        } catch (err: any) {
          alert(`✗ Error: ${err.message}`)
          // Revert decision on error
          if (imageType === 'original') {
            set.original.decision = 'undecided'
          } else {
            set.duplicate.decision = 'undecided'
          }
          setDuplicateSets([...newSets])
        } finally {
          setDeleting(false)
        }
      }
    },
    [currentIndex, sets]
  )

  const handlePrevious = useCallback(() => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1)
    }
  }, [currentIndex])

  const handleNext = useCallback(() => {
    if (currentIndex < sets.length - 1) {
      setCurrentIndex(currentIndex + 1)
    }
  }, [currentIndex, sets.length])

  const handleKeepOriginal = useCallback(() => {
    handleUpdateDecision('original', 'keep')
  }, [handleUpdateDecision])

  const handleKeepDuplicate = useCallback(() => {
    handleUpdateDecision('duplicate', 'keep')
  }, [handleUpdateDecision])

  const handleDeleteBoth = useCallback(async () => {
    const currentSet = sets[currentIndex]
    if (!currentSet) return

    if (!confirm('⚠️ Delete BOTH original and duplicate? This cannot be undone!')) {
      return
    }

    setDeleting(true)
    try {
      const response = await fetch('http://localhost:5000/api/delete-both', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          original: currentSet.original.path,
          duplicate: currentSet.duplicate.path,
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'Failed to delete both files')
      }

      const result = await response.json()
      console.log(result.message)

      // Remove this pair from the list (both files deleted)
      const updatedSets = sets.filter((_, idx) => idx !== currentIndex)
      setDuplicateSets(updatedSets)
      
      // Adjust index if needed
      if (currentIndex >= updatedSets.length && updatedSets.length > 0) {
        setCurrentIndex(updatedSets.length - 1)
      }
    } catch (err: any) {
      alert(`✗ Error: ${err.message}`)
    } finally {
      setDeleting(false)
    }
  }, [currentIndex, sets, reloadDuplicates])

  useKeyboard({
    onLeft: handlePrevious,
    onRight: handleNext,
    onOne: handleKeepOriginal,
    onTwo: handleKeepDuplicate,
    onThree: handleDeleteBoth,
  })

  const handleExport = useCallback(async () => {
    const undecidedCount = sets.filter(
      (set) => set.original.decision === 'undecided' && set.duplicate.decision === 'undecided'
    ).length

    if (undecidedCount > 0) {
      if (!confirm(`${undecidedCount} sets are still undecided. Export anyway?`)) {
        return
      }
    }

    const exportData: ExportData = {
      timestamp: new Date().toISOString(),
      totalSets: sets.length,
      reviewed: reviewedCount,
      decisions: sets
        .filter((set) => set.original.decision !== 'undecided' || set.duplicate.decision !== 'undecided')
        .map((set) => ({
          original: set.original.path,
          duplicate: set.duplicate.path,
          keepOriginal: set.original.decision === 'keep',
          keepDuplicate: set.duplicate.decision === 'keep',
        })),
    }

    try {
      const result = await exportDecisions(exportData)
      alert(`✓ Decisions exported to: ${result.filename}`)
    } catch (err: any) {
      alert(`✗ Error exporting: ${err.message}`)
    }
  }, [sets, reviewedCount])

  if (loading) {
    return (
      <div className="loading">
        <h1>⚡ Loading duplicates...</h1>
        <p>Fast mode: Metadata loaded on-demand</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="error">
        <h1>Error</h1>
        <p>{error}</p>
        <p>Make sure the Flask server is running: <code>python backend/server.py</code></p>
      </div>
    )
  }

  if (sets.length === 0) {
    return (
      <div className="empty">
        <h1>No Duplicates Found</h1>
        <p>Run the photo organizer script to generate duplicates.csv</p>
      </div>
    )
  }

  const hasConflict =
    currentSet &&
    currentSet.original.decision === 'keep' &&
    currentSet.duplicate.decision === 'keep'

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div>
            <h1>Photo Duplicate Checker</h1>
            <div className="header-info">
              <span>Set {currentIndex + 1} of {sets.length}</span>
              <span>•</span>
              <span className="badge">{reviewedCount} Reviewed</span>
              <span className="badge outline">{sets.length - reviewedCount} Remaining</span>
            </div>
          </div>
          
          <div className="header-actions">
            <button onClick={handleExport} className="btn-outline">
              Export Decisions
            </button>
            <button onClick={handlePrevious} disabled={currentIndex === 0} className="btn-outline">
              ←
            </button>
            <button onClick={handleNext} disabled={currentIndex >= sets.length - 1} className="btn-outline">
              →
            </button>
          </div>
        </div>
        
        <div className="keyboard-hints">
          <span>⌨️ Keyboard:</span>
          <code>←→</code> Navigate
          <code>1</code> Keep Original
          <code>2</code> Keep Duplicate
          <code>3</code> Delete Both
        </div>
      </header>

      <main className="main">
        {hasConflict && (
          <div className="alert alert-warning">
            ⚠️ Both images are marked to keep. Typically, you should delete one.
          </div>
        )}

        {deleting && (
          <div className="alert alert-info">
            🗑️ Deleting file...
          </div>
        )}

        {loadingMetadata && (
          <div className="alert alert-info">
            📊 Loading metadata...
          </div>
        )}

        {currentSet && (
          <>
            <div className="comparison">
              <ImageCard
                image={currentSet.original}
                label="Original"
                keyboardHint="Press 1"
                onKeep={() => handleUpdateDecision('original', 'keep')}
                onDelete={() => handleUpdateDecision('original', 'delete')}
              />
              <ImageCard
                image={currentSet.duplicate}
                label="Duplicate"
                keyboardHint="Press 2"
                onKeep={() => handleUpdateDecision('duplicate', 'keep')}
                onDelete={() => handleUpdateDecision('duplicate', 'delete')}
              />
            </div>
            
            <div style={{ textAlign: 'center', marginTop: '1rem' }}>
              <button
                onClick={handleDeleteBoth}
                className="btn btn-delete"
                disabled={deleting || (!currentSet.original.exists && !currentSet.duplicate.exists)}
                style={{ fontSize: '1rem', padding: '0.75rem 1.5rem' }}
              >
                🗑🗑 Delete Both (Press 3)
              </button>
            </div>
          </>
        )}
      </main>
    </div>
  )
}

function ImageCard({ image, label, keyboardHint, onKeep, onDelete }: any) {
  const [imageError, setImageError] = useState(false)
  const [imageLoaded, setImageLoaded] = useState(false)

  const getBorderColor = () => {
    if (!image.exists) return 'border-deleted'
    switch (image.decision) {
      case 'keep': return 'border-keep'
      case 'delete': return 'border-delete'
      default: return ''
    }
  }

  const getDecisionBadge = () => {
    if (!image.exists) return <span className="badge badge-deleted">Deleted</span>
    switch (image.decision) {
      case 'keep': return <span className="badge badge-keep">Keep</span>
      case 'delete': return <span className="badge badge-delete">Delete</span>
      default: return <span className="badge">Undecided</span>
    }
  }

  return (
    <div className={`card ${getBorderColor()}`}>
      <div className="image-container">
        {!imageLoaded && !imageError && <div className="image-loading" />}
        
        {imageError || !image.exists ? (
          <div className="image-error">
            <span>📷</span>
            <p>Image not found</p>
          </div>
        ) : (
          <img
            src={`http://localhost:5000/api/image?path=${encodeURIComponent(image.path)}`}
            alt={label}
            className={imageLoaded ? 'loaded' : ''}
            onLoad={() => setImageLoaded(true)}
            onError={() => setImageError(true)}
          />
        )}

        <div className="image-badges">
          <span className="badge badge-label">{label}</span>
          {keyboardHint && <span className="badge badge-kbd">{keyboardHint}</span>}
        </div>

        <div className="image-decision">
          {getDecisionBadge()}
        </div>
      </div>

      <div className="card-content">
        <div className="metadata">
          <div className="metadata-item">
            <label>Path</label>
            <p className="path" title={image.path}>{image.path}</p>
          </div>

          <div className="metadata-row">
            <div className="metadata-item">
              <label>Resolution</label>
              <p>{image.width && image.height ? `${image.width}×${image.height}` : (image.resolution ? `${(image.resolution / 1000000).toFixed(1)} MP` : 'Unknown')}</p>
            </div>
            <div className="metadata-item">
              <label>File Size</label>
              <p>{image.fileSize ? `${image.fileSize} MB` : 'Unknown'}</p>
            </div>
          </div>

          {image.dateTaken && (
            <div className="metadata-item">
              <label>Date Taken</label>
              <p>{image.dateTaken}</p>
            </div>
          )}

          {image.sha256 && (
            <div className="metadata-item">
              <label>SHA-256</label>
              <p className="hash">{image.sha256}</p>
            </div>
          )}
        </div>

        <div className="card-actions">
          <button
            onClick={onKeep}
            className={`btn ${image.decision === 'keep' ? 'btn-keep' : 'btn-outline'}`}
            disabled={!image.exists}
          >
            ✓ Keep
          </button>
          <button
            onClick={onDelete}
            className={`btn ${image.decision === 'delete' ? 'btn-delete' : 'btn-outline'}`}
            disabled={!image.exists}
          >
            🗑 Delete
          </button>
        </div>
      </div>
    </div>
  )
}

export default App
