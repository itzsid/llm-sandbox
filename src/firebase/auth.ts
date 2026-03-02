import type { User } from 'firebase/auth'
import { getFirebaseApp } from './config'

export type { User } from 'firebase/auth'

export async function signInWithGoogle(): Promise<User> {
  const app = await getFirebaseApp()
  const { getAuth, signInWithPopup, signInWithRedirect, GoogleAuthProvider } = await import('firebase/auth')
  const auth = getAuth(app)
  const provider = new GoogleAuthProvider()
  try {
    const result = await signInWithPopup(auth, provider)
    return result.user
  } catch (err: unknown) {
    // Fallback to redirect if popup is blocked
    if (err && typeof err === 'object' && 'code' in err && (err as { code: string }).code === 'auth/popup-blocked') {
      await signInWithRedirect(auth, provider)
      // After redirect, the auth state listener in useAuth will pick up the user
      throw new Error('Redirecting to sign-in page...')
    }
    throw err
  }
}

export async function logOut(): Promise<void> {
  const app = await getFirebaseApp()
  const { getAuth, signOut } = await import('firebase/auth')
  const auth = getAuth(app)
  await signOut(auth)
}

export async function subscribeAuthState(
  callback: (user: User | null) => void,
): Promise<() => void> {
  const app = await getFirebaseApp()
  const { getAuth, onAuthStateChanged } = await import('firebase/auth')
  const auth = getAuth(app)
  return onAuthStateChanged(auth, callback)
}
