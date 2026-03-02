import { useState, useEffect } from 'react'
import type { User } from 'firebase/auth'
import { isFirebaseConfigured } from '../firebase/config'
import { subscribeAuthState } from '../firebase/auth'

export function useAuth(): { user: User | null; loading: boolean } {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(() => isFirebaseConfigured())

  useEffect(() => {
    if (!isFirebaseConfigured()) return

    let cancelled = false
    let unsubscribe: (() => void) | undefined

    subscribeAuthState((u) => {
      if (!cancelled) {
        setUser(u)
        setLoading(false)
      }
    }).then((unsub) => {
      if (cancelled) {
        unsub() // cleanup already ran, unsubscribe immediately
      } else {
        unsubscribe = unsub
      }
    }).catch(() => {
      if (!cancelled) setLoading(false)
    })

    return () => {
      cancelled = true
      unsubscribe?.()
    }
  }, [])

  return { user, loading }
}
