import { useState } from 'react'
import './App.css';
import Det from './Det.jsx';
function App() {
  const [count, setCount] = useState(0)

  return (
    <>
        <Det></Det>
    </>
  );
}

export default App
