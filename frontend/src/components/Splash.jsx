import { motion } from 'framer-motion'
import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

export default function Splash() {
  const navigate = useNavigate()

  useEffect(() => {
    const timer = setTimeout(() => navigate('/explicacao'), 4000)
    return () => clearTimeout(timer)
  }, [navigate])

  return (
    <motion.div 
      className="w-full h-full flex flex-col items-center justify-center cursor-pointer px-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, y: -30 }}
      transition={{ duration: 0.8 }}
      onClick={() => navigate('/explicacao')}
    >
      <motion.div 
        className="w-full max-w-3xl px-4 sm:px-8 flex justify-center items-center mb-6 sm:mb-10"
        initial={{ opacity: 0, scale: 0.95, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
      >
        <img 
          src="/logo.png" 
          alt="Okuyeva Logo" 
          className="w-full max-w-xl max-h-[55vh] object-contain drop-shadow-sm"
        />
      </motion.div>

      
      <motion.div 
        className="mt-10 sm:mt-16 flex items-center gap-3"
        style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.2em', color: 'var(--text-mono)' }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 1 }}
      >
        <motion.div 
          className="h-[2px] rounded-full"
          style={{ background: 'var(--accent-blue)' }}
          initial={{ width: 0 }}
          animate={{ width: 28 }}
          transition={{ delay: 2, duration: 0.8, ease: "circOut" }}
        />
        <motion.span
          animate={{ opacity: [0.6, 1, 0.6] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        >
          Tocar para continuar
        </motion.span>
      </motion.div>
    </motion.div>
  )
}
