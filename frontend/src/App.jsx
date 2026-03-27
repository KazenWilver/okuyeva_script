import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Splash from './components/Splash'
import Explicacoes from './components/Explicacoes'
import MenuPrincipal from './components/MenuPrincipal'
import Consulta from './components/Consulta'
import Informacoes from './components/Informacoes'

function App() {
  return (
    <Router>
      <div className="w-screen h-screen overflow-hidden bg-gradient-to-br from-teal-50 to-emerald-50 flex items-center justify-center">
        <div className="w-full h-full sm:p-4 md:p-8">
          <div className="w-full h-full bg-white/60 backdrop-blur-2xl sm:rounded-[2.5rem] shadow-2xl overflow-hidden relative border border-white/50">
            <AnimatePresence mode="wait">
              <Routes>
              <Route path="/" element={<Splash />} />
              <Route path="/explicacao" element={<Explicacoes />} />
              <Route path="/menu" element={<MenuPrincipal />} />
              <Route path="/consulta" element={<Consulta />} />
              <Route path="/informacoes" element={<Informacoes />} />
            </Routes>
          </AnimatePresence>
        </div>
      </div>
    </div>
  </Router>
)
}

export default App
