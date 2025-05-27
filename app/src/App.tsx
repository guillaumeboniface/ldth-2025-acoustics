import React, { useEffect } from 'react'
import Recorder from './components/Recorder'
import init from 'rust-melspec-wasm'
import './App.css'

function App() {
    useEffect(() => {
        init().then(() => {
            console.log("Melspec initialized");
        });
        }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Drone detection</h1>
      </header>
      <main>
        <Recorder />
      </main>
    </div>
  )
}

export default App