import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, BookOpen, Shield, HelpCircle } from 'lucide-react'

const cards = [
  {
    icon: BookOpen,
    color: "#34d399",
    bg: "rgba(52,211,153,0.1)",
    label: "Funcionamento",
    title: "Como funciona?",
    body: "Os seus padrões corporais são lidos e processados em milissegundos, gerando texto compreensível de forma fluída durante toda a consulta.",
  },
  {
    icon: Shield,
    color: "#60a5fa",
    bg: "rgba(96,165,250,0.1)",
    label: "Privacidade",
    title: "Dados Seguros",
    body: "Não gravamos imagens na rede. O nosso motor baseia-se num sistema de metadados corporais instantâneos. Tudo é processado localmente.",
  },
  {
    icon: HelpCircle,
    color: "#2dd4bf",
    bg: "rgba(45,212,191,0.1)",
    label: "Suporte",
    title: "Ajuda Técnica",
    body: "Para questões em relação a hardware, ou falta de câmara, fale com a recepção principal.",
  },
]

export default function Informacoes() {
  const navigate = useNavigate()

  return (
    <motion.div 
      className="w-full h-full flex flex-col px-4 sm:px-6 py-4 sm:py-6 overflow-y-auto"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Header */}
      <div className="flex items-center gap-3 mb-6 sm:mb-8 flex-shrink-0">
        <button 
          onClick={() => navigate('/menu')}
          className="strata-card w-9 h-9 sm:w-10 sm:h-10 flex items-center justify-center cursor-pointer !rounded-full"
        >
          <ArrowLeft size={15} />
        </button>
        <span className="mono-tag">Informações</span>
      </div>

      <div className="max-w-2xl mx-auto w-full">
        <h1 className="display-text mb-4">Alguma<br/>Dúvida?</h1>

        <p className="text-sm sm:text-base max-w-md mb-8 leading-relaxed" style={{ color: 'var(--text-mono)' }}>
          O Okuyeva aproxima pacientes surdos e médicos através de tradução instantânea com recurso à IA local.
        </p>

        <div className="flex flex-col gap-4">
          {cards.map((card, i) => (
            <motion.div
              key={i}
              className="strata-card p-6 sm:p-8"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + i * 0.1 }}
            >
              <div className="flex gap-4 items-start">
                <div className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: card.bg }}>
                  <card.icon size={18} strokeWidth={1.4} style={{ color: card.color }} />
                </div>
                <div className="flex-1 min-w-0">
                  <span className="mono-label block mb-1">{card.label}</span>
                  <h3 className="text-base sm:text-lg font-extrabold tracking-tight mb-2">{card.title}</h3>
                  <p className="text-xs sm:text-sm leading-relaxed" style={{ color: 'var(--text-mono)' }}>{card.body}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        <div className="mt-8 flex items-center gap-3">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          <span className="mono-label">v2.0 · LSTM · Hackathon Okuyeva · Angola 2026</span>
        </div>
      </div>
    </motion.div>
  )
}
