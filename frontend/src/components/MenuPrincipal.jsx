import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { Stethoscope, Info } from 'lucide-react'

export default function MenuPrincipal() {
  const navigate = useNavigate()

  return (
    <motion.div 
      className="w-full h-full flex flex-col items-center justify-center px-5 sm:px-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
    >
      <motion.span 
        className="mono-tag mb-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1 }}
      >
        Menu Principal
      </motion.span>

      <motion.h1 
        className="display-text text-center mb-10 sm:mb-14"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        O que deseja<br/>fazer?
      </motion.h1>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-5 w-full max-w-lg">
        <motion.button
          onClick={() => navigate('/consulta')}
          className="strata-card p-7 sm:p-8 text-left cursor-pointer"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          whileTap={{ scale: 0.97 }}
        >
          <div className="w-10 h-10 rounded-full flex items-center justify-center mb-5" style={{ background: 'rgba(52,211,153,0.1)' }}>
            <Stethoscope size={20} strokeWidth={1.4} className="text-emerald-500" />
          </div>
          <div className="text-lg sm:text-xl font-extrabold tracking-tight">Consulta</div>
          <p className="text-xs sm:text-sm mt-2 leading-relaxed" style={{ color: 'var(--text-mono)' }}>
            Tradução gestual em tempo real
          </p>
        </motion.button>

        <motion.button
          onClick={() => navigate('/informacoes')}
          className="strata-card p-7 sm:p-8 text-left cursor-pointer"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          whileTap={{ scale: 0.97 }}
        >
          <div className="w-10 h-10 rounded-full flex items-center justify-center mb-5" style={{ background: 'rgba(96,165,250,0.1)' }}>
            <Info size={20} strokeWidth={1.4} className="text-blue-400" />
          </div>
          <div className="text-lg sm:text-xl font-extrabold tracking-tight">Informações</div>
          <p className="text-xs sm:text-sm mt-2 leading-relaxed" style={{ color: 'var(--text-mono)' }}>
            Como funciona o sistema
          </p>
        </motion.button>
      </div>

      <motion.div 
        className="mt-10 sm:mt-14 flex items-center gap-3"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
      >
        <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 pulse-ring" />
        <span className="mono-label"></span>
      </motion.div>
    </motion.div>
  )
}
