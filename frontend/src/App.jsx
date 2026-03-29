import { AnimatePresence } from 'framer-motion'
import { Routes, Route, useLocation } from 'react-router-dom'
import Splash from './components/Splash'
import Explicacoes from './components/Explicacoes'
import MenuPrincipal from './components/MenuPrincipal'
import Consulta from './components/Consulta'
import Informacoes from './components/Informacoes'

export default function App() {
  const location = useLocation()

  return (
    <div className="w-screen h-screen relative overflow-hidden" style={{ backgroundColor: '#f2f2f2' }}>
      <div className="grain" />
      <div className="relative z-10 w-full h-full">
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<Splash />} />
            <Route path="/explicacao" element={<Explicacoes />} />
            <Route path="/menu" element={<MenuPrincipal />} />
            <Route path="/consulta" element={<Consulta />} />
            <Route path="/informacoes" element={<Informacoes />} />
          </Routes>
        </AnimatePresence>
      </div>
    </div>
  )
}
