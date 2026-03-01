import { useState, useEffect } from 'react'
import type { User } from 'firebase/auth'
import { isFirebaseConfigured } from '../firebase/config'
import { subscribeAuthState } from '../firebase/auth'

export function useAuth(): { user: User | null; loading: boolean } {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(() => isFirebaseConfigured())

  useEffect(() => {
    if (!isFirebaseConfigured()) return

    let unsubscribe: (() => void) | null = null

    subscribeAuthState((u) => {
      setUser(u)
      setLoading(false)
    }).then((unsub) => {
      unsubscribe = unsub
    }).catch(() => {
      setLoading(false)
    })

    return () => {
      unsubscribe?.()
    }
  }, [])

  return { user, loading }
}
