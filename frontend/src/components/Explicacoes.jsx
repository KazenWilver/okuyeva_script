import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ChevronRight, ChevronLeft, Check } from 'lucide-react'

const slides = [
  {
    tag: "01 · Introdução",
    title: "Bem-vindo",
    body: "O Okuyeva utiliza Inteligência Artificial para traduzir Língua Gestual Angolana directamente em texto clínico.",
  },
  {
    tag: "02 · Calibração",
    title: "Posição",
    body: "Certifique-se de que a câmara capta claramente o seu rosto e as suas duas mãos para uma tradução mais fiável.",
  },
  {
    tag: "03 · Activo",
    title: "Pronto",
    body: "Pode comunicar as suas dores e sintomas livremente. O seu médico lerá as transcrições exactas em tempo real.",
  },
]

export default function Explicacoes() {
  const [index, setIndex] = useState(0)
  const navigate = useNavigate()

  const next = () => (index < slides.length - 1 ? setIndex(index + 1) : navigate('/menu'))
  const prev = () => index > 0 && setIndex(index - 1)

  return (
    <motion.div 
      className="w-full h-full flex items-center justify-center px-5 sm:px-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="strata-card w-full max-w-2xl p-8 sm:p-14 lg:p-16">
        <AnimatePresence mode="wait">
          <motion.div
            key={index}
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -30 }}
            transition={{ duration: 0.25 }}
          >
            <span className="mono-tag mb-6">{slides[index].tag}</span>

            <h2 className="display-text mt-5 mb-5">{slides[index].title}</h2>

            <p className="text-sm sm:text-base max-w-md leading-relaxed" style={{ color: 'var(--text-mono)' }}>
              {slides[index].body}
            </p>
          </motion.div>
        </AnimatePresence>

        {/* Navigation */}
        <div className="flex items-center justify-between mt-12 sm:mt-16">
          <button 
            onClick={prev}
            className={`strata-card w-10 h-10 sm:w-11 sm:h-11 flex items-center justify-center cursor-pointer !rounded-full transition-opacity ${
              index === 0 ? 'opacity-20 !cursor-not-allowed' : ''
            }`}
            disabled={index === 0}
          >
            <ChevronLeft size={16} />
          </button>

          <div className="flex gap-2.5">
            {slides.map((_, i) => (
              <div 
                key={i} 
                className="h-[2.5px] rounded-full transition-all duration-500"
                style={{
                  width: i === index ? '32px' : '10px',
                  background: i === index ? 'var(--accent-green)' : 'rgba(0,0,0,0.1)',
                }}
              />
            ))}
          </div>

          <button 
            onClick={next}
            className="w-10 h-10 sm:w-11 sm:h-11 flex items-center justify-center cursor-pointer rounded-full text-white transition-all hover:-translate-y-0.5"
            style={{ 
              background: index === slides.length - 1 ? 'var(--accent-green)' : 'var(--text-solid)',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            }}
          >
            {index === slides.length - 1 ? <Check size={16} /> : <ChevronRight size={16} />}
          </button>
        </div>
      </div>
    </motion.div>
  )
}
