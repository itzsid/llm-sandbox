import type { User } from 'firebase/auth'
import { getFirebaseApp } from './config'

export type { User } from 'firebase/auth'

export async function signInWithGoogle(): Promise<User> {
  const app = await getFirebaseApp()
  const { getAuth, signInWithPopup, GoogleAuthProvider } = await import('firebase/auth')
  const auth = getAuth(app)
  const result = await signInWithPopup(auth, new GoogleAuthProvider())
  return result.user
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
