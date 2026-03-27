import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { Stethoscope, Info } from 'lucide-react'

export default function MenuPrincipal() {
  const navigate = useNavigate()

  const cardVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: { opacity: 1, y: 0 }
  }

  return (
    <motion.div 
      className="w-full h-full flex flex-col p-8 items-center justify-center bg-white"
      initial="hidden"
      animate="visible"
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ staggerChildren: 0.2 }}
    >
      <motion.h1 
        className="text-4xl lg:text-5xl font-extrabold text-slate-800 mb-16 text-center"
        variants={cardVariants}
      >
        O que deseja fazer hoje?
      </motion.h1>

      <div className="flex flex-col sm:flex-row gap-8 lg:gap-16 w-full max-w-5xl justify-center items-center">
        {/* Consulta Button */}
        <motion.button
          variants={cardVariants}
          whileHover={{ scale: 1.05, translateY: -10 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => navigate('/consulta')}
          className="w-full max-w-sm aspect-square bg-gradient-to-br from-teal-400 to-emerald-500 rounded-[3rem] p-8 flex flex-col items-center justify-center text-white shadow-2xl shadow-teal-500/40 group relative overflow-hidden"
        >
          <div className="absolute inset-0 bg-white/20 group-hover:opacity-0 transition-opacity blur-2xl rounded-full scale-150 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"></div>
          <motion.div 
            className="bg-white/20 p-8 rounded-full mb-8 backdrop-blur-md"
            animate={{ y: [0, -10, 0] }}
            transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
          >
            <Stethoscope size={80} strokeWidth={1.5} />
          </motion.div>
          <span className="text-3xl font-bold tracking-wide">Consulta</span>
        </motion.button>

        {/* Informacoes Button */}
        <motion.button
          variants={cardVariants}
          whileHover={{ scale: 1.05, translateY: -10 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => navigate('/informacoes')}
          className="w-full max-w-sm aspect-square bg-gradient-to-br from-indigo-400 to-purple-600 rounded-[3rem] p-8 flex flex-col items-center justify-center text-white shadow-2xl shadow-purple-500/40 group relative overflow-hidden"
        >
          <div className="absolute inset-0 bg-white/20 group-hover:opacity-0 transition-opacity blur-2xl rounded-full scale-150 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"></div>
          <div className="bg-white/20 p-8 rounded-full mb-8 backdrop-blur-md">
            <Info size={80} strokeWidth={1.5} />
          </div>
          <span className="text-3xl font-bold tracking-wide">Informações</span>
        </motion.button>
      </div>

    </motion.div>
  )
}
