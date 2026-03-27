import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ChevronRight, ChevronLeft, Check } from 'lucide-react'

const slides = [
  {
    title: "Bem-vindo às Explicações",
    content: "O Zero Barreiras utiliza Inteligência Artificial para traduzir Língua Gestual Angolana directamente em texto clínico."
  },
  {
    title: "Como Posicionar-se?",
    content: "Certifique-se de que a câmara capta claramente o seu rosto e as suas duas mãos para uma tradução mais fiável."
  },
  {
    title: "Tudo Pronto!",
    content: "Pode comunicar as suas dores e sintomas livremente. O seu médico lerá as transcrições exactas em tempo real."
  }
]

export default function Explicacoes() {
  const [index, setIndex] = useState(0)
  const navigate = useNavigate()

  const handleNext = () => {
    if (index < slides.length - 1) setIndex(index + 1)
    else navigate('/menu')
  }

  const handlePrev = () => {
    if (index > 0) setIndex(index - 1)
  }

  return (
    <motion.div 
      className="w-full h-full flex flex-col items-center justify-center p-8 lg:p-24"
      initial={{ opacity: 0, x: 100 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -100 }}
      transition={{ duration: 0.5, ease: "anticipate" }}
    >
      <div className="w-full max-w-4xl bg-white/80 backdrop-blur-3xl shadow-2xl shadow-slate-200/50 rounded-[3rem] p-12 lg:p-20 border border-white relative overflow-hidden h-[500px] lg:h-[600px] flex flex-col">
        
        <div className="flex-1 relative">
          <AnimatePresence mode="wait">
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="absolute inset-0 flex flex-col justify-center items-center text-center px-4 lg:px-12"
            >
              <h2 className="text-4xl lg:text-5xl font-extrabold text-slate-800 mb-8">{slides[index].title}</h2>
              <p className="text-slate-600 text-2xl lg:text-3xl leading-relaxed">{slides[index].content}</p>
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Carousel controls */}
        <div className="flex items-center justify-between mt-12">
          <button 
            onClick={handlePrev}
            className={`p-5 rounded-full transition-colors ${index === 0 ? 'text-slate-300 cursor-not-allowed' : 'text-slate-600 hover:bg-slate-100'}`}
            disabled={index === 0}
          >
            <ChevronLeft size={36} />
          </button>
          
          <div className="flex gap-4">
            {slides.map((_, i) => (
              <div 
                key={i} 
                className={`h-3 rounded-full transition-all duration-300 ${i === index ? 'w-12 bg-sky-500' : 'w-3 bg-slate-200'}`}
              />
            ))}
          </div>

          <button 
            onClick={handleNext}
            className="p-5 bg-sky-500 text-white rounded-full hover:bg-sky-600 transition-all shadow-xl shadow-sky-500/30"
          >
            {index === slides.length - 1 ? <Check size={36} /> : <ChevronRight size={36} />}
          </button>
        </div>

      </div>
    </motion.div>
  )
}
