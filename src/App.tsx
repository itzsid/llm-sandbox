import { GpuStatus } from './ui/GpuStatus'
import { TrainingPanel } from './ui/TrainingPanel'
import './App.css'

function App() {
  return (
    <div className="app">
      <header className="header">
        <h1>LLM Sandbox</h1>
        <GpuStatus />
      </header>
      <main className="main">
        <TrainingPanel />
      </main>
    </div>
  )
}

export default App
