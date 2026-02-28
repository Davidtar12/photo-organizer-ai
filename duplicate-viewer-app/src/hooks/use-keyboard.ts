import { useEffect } from 'react'

export function useKeyboard(handlers: {
  onLeft?: () => void
  onRight?: () => void
  onOne?: () => void
  onTwo?: () => void
  onThree?: () => void
}) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault()
          handlers.onLeft?.()
          break
        case 'ArrowRight':
          e.preventDefault()
          handlers.onRight?.()
          break
        case '1':
          e.preventDefault()
          handlers.onOne?.()
          break
        case '2':
          e.preventDefault()
          handlers.onTwo?.()
          break
        case '3':
          e.preventDefault()
          handlers.onThree?.()
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handlers])
}
