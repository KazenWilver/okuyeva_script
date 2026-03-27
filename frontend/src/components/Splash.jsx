import { motion } from 'framer-motion'
import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { HeartPulse } from 'lucide-react'

export default function Splash() {
  const navigate = useNavigate()

  useEffect(() => {
    // Automatically transition to the next screen after 4 seconds
    const timer = setTimeout(() => {
      navigate('/explicacao')
    }, 4000)
    return () => clearTimeout(timer)
  }, [navigate])

  return (
    <motion.div 
      className="w-full h-full flex flex-col items-center justify-center cursor-pointer p-8 lg:p-24"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, y: -50 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      onClick={() => navigate('/explicacao')}
    >
      <motion.div 
        className="w-48 h-48 lg:w-64 lg:h-64 bg-gradient-to-tr from-sky-400 to-teal-400 rounded-full flex items-center justify-center shadow-2xl shadow-sky-400/40 mb-12"
        initial={{ rotate: -180, scale: 0 }}
        animate={{ rotate: 0, scale: 1 }}
        transition={{ type: 'spring', damping: 15, delay: 0.2 }}
      >
        <HeartPulse className="w-24 h-24 lg:w-32 lg:h-32 text-white" strokeWidth={1.5} />
      </motion.div>
      
      <motion.h1 
        className="text-5xl lg:text-7xl font-extrabold text-slate-800 tracking-tight text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.8 }}
      >
        Zero Barreiras
      </motion.h1>
      
      <motion.p 
        className="text-xl lg:text-3xl text-slate-500 mt-6 font-medium max-w-3xl text-center leading-relaxed"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1, duration: 1 }}
      >
        "A saúde e o bem-estar ao alcance de todos, sem excepção."
      </motion.p>
      
      <motion.div 
        className="mt-24 text-sm lg:text-base font-bold text-slate-400 uppercase tracking-[0.3em]"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2, duration: 1 }}
      >
        Tocar para continuar
      </motion.div>
    </motion.div>
  )
}
