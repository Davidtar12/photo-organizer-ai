import { useState } from 'react'
import { BulkView } from './components/BulkView'
import { ReviewView } from './components/ReviewView'

type ViewMode = 'bulk' | 'review'

function AppWrapper() {
  const [viewMode, setViewMode] = useState<ViewMode>('bulk')
  const [selectedPairIndex, setSelectedPairIndex] = useState<number>(0)

  const handleOpenReview = (index: number) => {
    setSelectedPairIndex(index)
    setViewMode('review')
  }

  const handleBackToBulk = () => {
    setViewMode('bulk')
  }

  if (viewMode === 'bulk') {
    return <BulkView onOpenReview={handleOpenReview} />
  }

  return <ReviewView initialIndex={selectedPairIndex} onBack={handleBackToBulk} />
}

export default AppWrapper
