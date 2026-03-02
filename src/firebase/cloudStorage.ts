import { getFirebaseApp } from './config'
import { exportCheckpoint, importCheckpointFromBuffer } from '../storage/checkpoint'
import type { Checkpoint } from '../storage/checkpoint'

export interface CloudCheckpointMeta {
  name: string
  step: number
  savedAt: number
  datasetId: string
  size: number
  configSummary: string
}

function configSummary(config: Checkpoint['config']): string {
  return `${config.nLayers}L/${config.nHeads}H/${config.dModel}d`
}

// Firestore max document size is ~1MB. We use 800KB chunks to leave room for metadata overhead.
const CHUNK_SIZE = 800 * 1024

export async function saveCloudCheckpoint(
  uid: string,
  checkpoint: Checkpoint,
): Promise<void> {
  const app = await getFirebaseApp()
  const { getFirestore, doc, setDoc, collection, getDocs, writeBatch, Bytes } = await import('firebase/firestore')

  const blob = exportCheckpoint(checkpoint)
  const buffer = await blob.arrayBuffer()
  const bytes = new Uint8Array(buffer)
  const totalChunks = Math.ceil(bytes.length / CHUNK_SIZE)

  const db = getFirestore(app)
  const chunksCol = collection(db, 'users', uid, 'checkpoints', checkpoint.name, 'chunks')

  // Delete any existing chunks (handles overwrite case)
  const existingChunks = await getDocs(chunksCol)
  if (existingChunks.size > 0) {
    const docsToDelete = existingChunks.docs
    for (let i = 0; i < docsToDelete.length; i += 500) {
      const batch = writeBatch(db)
      const slice = docsToDelete.slice(i, i + 500)
      slice.forEach((d) => batch.delete(d.ref))
      await batch.commit()
    }
  }

  // Write data chunks first (before metadata)
  for (let i = 0; i < totalChunks; i++) {
    const start = i * CHUNK_SIZE
    const end = Math.min(start + CHUNK_SIZE, bytes.length)
    const chunk = bytes.slice(start, end)
    const chunkDoc = doc(chunksCol, String(i))
    const batch = writeBatch(db)
    batch.set(chunkDoc, { data: Bytes.fromUint8Array(chunk), index: i })
    await batch.commit()
  }

  // Write metadata doc last — this is the "commit point"
  const metaDoc = doc(db, 'users', uid, 'checkpoints', checkpoint.name)
  await setDoc(metaDoc, {
    name: checkpoint.name,
    step: checkpoint.step,
    savedAt: checkpoint.savedAt,
    datasetId: checkpoint.datasetId,
    size: buffer.byteLength,
    totalChunks,
    configSummary: configSummary(checkpoint.config),
  })
}

export async function listCloudCheckpoints(
  uid: string,
): Promise<CloudCheckpointMeta[]> {
  const app = await getFirebaseApp()
  const { getFirestore, collection, query, orderBy, getDocs } = await import('firebase/firestore')

  const db = getFirestore(app)
  const q = query(
    collection(db, 'users', uid, 'checkpoints'),
    orderBy('savedAt', 'desc'),
  )
  const snap = await getDocs(q)
  return snap.docs.map((d) => d.data() as CloudCheckpointMeta)
}

export async function loadCloudCheckpoint(
  uid: string,
  name: string,
): Promise<Checkpoint> {
  const app = await getFirebaseApp()
  const { getFirestore, doc, getDoc, collection, query, orderBy, getDocs } = await import('firebase/firestore')

  const db = getFirestore(app)

  // Read metadata to get totalChunks
  const metaDoc = doc(db, 'users', uid, 'checkpoints', name)
  const metaSnap = await getDoc(metaDoc)
  if (!metaSnap.exists()) throw new Error(`Checkpoint "${name}" not found`)
  const meta = metaSnap.data() as CloudCheckpointMeta & { totalChunks: number }

  // Read all chunks
  const chunksCol = collection(db, 'users', uid, 'checkpoints', name, 'chunks')
  const q = query(chunksCol, orderBy('index', 'asc'))
  const chunkSnap = await getDocs(q)

  // Reassemble buffer
  const totalSize = meta.size
  const assembled = new Uint8Array(totalSize)
  let offset = 0
  chunkSnap.docs.forEach((d) => {
    const chunkBytes: Uint8Array = d.data().data.toUint8Array()
    assembled.set(chunkBytes, offset)
    offset += chunkBytes.length
  })

  // Validate reassembly
  if (chunkSnap.size !== meta.totalChunks) {
    throw new Error(`Checkpoint "${name}" is corrupted: expected ${meta.totalChunks} chunks but found ${chunkSnap.size}`)
  }
  if (offset !== totalSize) {
    throw new Error(`Checkpoint "${name}" is corrupted: expected ${totalSize} bytes but reassembled ${offset}`)
  }

  return importCheckpointFromBuffer(assembled.buffer)
}

export async function deleteCloudCheckpoint(
  uid: string,
  name: string,
): Promise<void> {
  const app = await getFirebaseApp()
  const { getFirestore, doc, deleteDoc, collection, getDocs, writeBatch } = await import('firebase/firestore')

  const db = getFirestore(app)

  // Delete all chunks first
  const chunksCol = collection(db, 'users', uid, 'checkpoints', name, 'chunks')
  const chunkSnap = await getDocs(chunksCol)
  if (chunkSnap.size > 0) {
    const batch = writeBatch(db)
    chunkSnap.docs.forEach((d) => batch.delete(d.ref))
    await batch.commit()
  }

  // Delete metadata doc
  const metaDoc = doc(db, 'users', uid, 'checkpoints', name)
  await deleteDoc(metaDoc)
}
