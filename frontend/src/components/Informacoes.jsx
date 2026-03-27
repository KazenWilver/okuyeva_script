import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, BookOpen, Shield, HelpCircle } from 'lucide-react'

export default function Informacoes() {
  const navigate = useNavigate()

  return (
    <motion.div 
      className="w-full h-full flex flex-col bg-white overflow-y-auto"
      initial={{ opacity: 0, y: '50%' }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: '50%' }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
    >
      <div className="h-40 lg:h-64 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-b-[3rem] lg:rounded-b-[5rem] p-8 lg:p-16 text-white relative flex items-end shadow-2xl">
        <button 
          onClick={() => navigate('/menu')}
          className="absolute top-8 left-8 lg:top-12 lg:left-12 p-3 lg:p-4 rounded-full bg-white/20 hover:bg-white/30 backdrop-blur-md transition-colors"
        >
          <ArrowLeft size={32} />
        </button>
        <h1 className="text-4xl lg:text-6xl font-extrabold ml-4">Alguma Dúvida?</h1>
      </div>

      <div className="p-10 lg:p-24 flex flex-col gap-10 lg:gap-14 max-w-6xl mx-auto w-full">
        <p className="text-slate-600 leading-relaxed font-medium text-xl lg:text-3xl">
          O Zero Barreiras é uma solução construída para aproximar pacientes mudos e médicos através de tradução instantânea com recurso à IA local.
        </p>

        <div className="flex gap-6 p-8 lg:p-10 rounded-3xl bg-indigo-50 border border-indigo-100 shadow-sm">
          <BookOpen className="text-indigo-500 shrink-0 w-12 h-12 lg:w-16 lg:h-16" />
          <div className="flex-1">
            <h3 className="font-bold text-indigo-900 text-2xl lg:text-3xl">Como funciona?</h3>
            <p className="text-indigo-700 text-lg lg:text-xl mt-3 leading-relaxed">Os seus padrões corporais são lidos e processados em milissegundos, gerando texto compreensível de forma fluída durante toda a consulta.</p>
          </div>
        </div>

        <div className="flex gap-6 p-8 lg:p-10 rounded-3xl bg-emerald-50 border border-emerald-100 shadow-sm">
          <Shield className="text-emerald-500 shrink-0 w-12 h-12 lg:w-16 lg:h-16" />
          <div className="flex-1">
            <h3 className="font-bold text-emerald-900 text-2xl lg:text-3xl">Privacidade Garantida</h3>
            <p className="text-emerald-700 text-lg lg:text-xl mt-3 leading-relaxed">Não gravamos imagens na rede. O nosso motor baseia-se num sistema de metadados corporais instantâneos.</p>
          </div>
        </div>

        <div className="flex gap-6 p-8 lg:p-10 rounded-3xl bg-sky-50 border border-sky-100 mb-12 shadow-sm">
          <HelpCircle className="text-sky-500 shrink-0 w-12 h-12 lg:w-16 lg:h-16" />
          <div className="flex-1">
            <h3 className="font-bold text-sky-900 text-2xl lg:text-3xl">Suporte Médico</h3>
            <p className="text-sky-700 text-lg lg:text-xl mt-3 leading-relaxed">Para questões em relação a hardware, ou falta de câmara, puxe a cordinha de emergência e fale com recepção principal.</p>
          </div>
        </div>

      </div>
    </motion.div>
  )
}
