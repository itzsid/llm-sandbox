import type { User } from 'firebase/auth'
import { isFirebaseConfigured } from '../firebase/config'
import { signInWithGoogle, logOut } from '../firebase/auth'
import { useState } from 'react'

interface AuthButtonProps {
  user: User | null
  loading: boolean
}

export function AuthButton({ user, loading }: AuthButtonProps) {
  const [busy, setBusy] = useState(false)

  if (!isFirebaseConfigured()) return null
  if (loading) return <span style={styles.loading}>...</span>

  if (user) {
    return (
      <div style={styles.userInfo}>
        {user.photoURL && (
          <img src={user.photoURL} alt="" style={styles.avatar} referrerPolicy="no-referrer" />
        )}
        <span style={styles.userName}>{user.displayName ?? user.email}</span>
        <button
          onClick={async () => { setBusy(true); await logOut(); setBusy(false) }}
          disabled={busy}
          style={styles.signOutBtn}
        >
          Sign out
        </button>
      </div>
    )
  }

  return (
    <button
      onClick={async () => { setBusy(true); try { await signInWithGoogle() } finally { setBusy(false) } }}
      disabled={busy}
      style={styles.signInBtn}
    >
      {busy ? 'Signing in...' : 'Sign in with Google'}
    </button>
  )
}

const styles: Record<string, React.CSSProperties> = {
  loading: {
    color: '#888',
    fontSize: '0.8rem',
  },
  userInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  avatar: {
    width: 24,
    height: 24,
    borderRadius: '50%',
  },
  userName: {
    color: '#ccc',
    fontSize: '0.8rem',
    maxWidth: 120,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  signOutBtn: {
    padding: '0.25rem 0.5rem',
    background: '#444',
    border: 'none',
    borderRadius: '4px',
    color: '#ccc',
    fontSize: '0.75rem',
    cursor: 'pointer',
  },
  signInBtn: {
    padding: '0.35rem 0.75rem',
    background: '#4285f4',
    border: 'none',
    borderRadius: '4px',
    color: '#fff',
    fontSize: '0.8rem',
    cursor: 'pointer',
  },
}
