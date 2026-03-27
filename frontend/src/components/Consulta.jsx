import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Video, Activity } from 'lucide-react'
import { useState, useEffect } from 'react'

export default function Consulta() {
  const navigate = useNavigate()
  
  const [transcription, setTranscription] = useState("")
  const [isConnected, setIsConnected] = useState(false)
  
  // High Performance Transmission Loop - Pulling texts only
  useEffect(() => {
    const fetchState = async () => {
      try {
        const response = await fetch("http://localhost:8000/tracking_state")
        const data = await response.json()
        setTranscription(data.gesture || "")
        setIsConnected(data.camera_active)
      } catch (err) {
        setIsConnected(false)
      }
    }

    const intervalId = setInterval(fetchState, 100);

    return () => {
      clearInterval(intervalId);
    }
  }, [])

  return (
    <motion.div 
      className="min-h-screen bg-slate-50 flex flex-col items-center pb-12"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
    >
      <div className="w-full max-w-6xl px-4 sm:px-6 lg:px-8 mt-6">
        
        <header className="flex items-center justify-between bg-white rounded-3xl shadow-sm p-4 sm:p-6 mb-6">
          <div className="flex items-center gap-4">
            <button 
              onClick={() => navigate('/menu')}
              className="p-3 rounded-full hover:bg-slate-100 text-slate-500 transition-colors"
            >
              <ArrowLeft size={24} />
            </button>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-800">Em Consulta</h1>
          </div>
          <div className="flex items-center gap-2 bg-slate-50 px-4 py-2 rounded-full border border-slate-200">
            <div className={`w-3 h-3 rounded-full flex-shrink-0 ${isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} />
            <span className="text-sm font-bold text-slate-600 hidden sm:inline-block">
              {isConnected ? 'Hardware C++ Ativo' : 'A Ligar AI...'}
            </span>
          </div>
        </header>

        <main className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          <div className="lg:col-span-2 bg-black rounded-3xl overflow-hidden shadow-xl border-4 border-slate-800 relative aspect-video flex items-center justify-center">
            <img 
              src="http://localhost:8000/video_feed" 
              alt="Video Feed" 
              className="w-full h-full object-cover" 
              onError={(e) => { e.target.style.display = 'none'; }}
              onLoad={(e) => { e.target.style.display = 'block'; }}
              style={{ display: 'none' }}
            />
            {/* Fallback layout when image stream dies */}
            {!isConnected && (
              <div className="absolute text-slate-500 flex flex-col items-center">
                <Video size={48} className="mb-4 opacity-50 animate-pulse" />
                <p className="font-semibold text-lg text-center">A aguardar API Nativa do Python...</p>
              </div>
            )}
          </div>

          <div className="lg:col-span-1 flex flex-col gap-6">
            <div className="bg-white rounded-3xl p-6 shadow-sm border border-slate-100 flex-1 flex flex-col justify-center">
              <div className="flex items-center gap-2 mb-4">
                <Activity size={20} className="text-sky-500 animate-pulse" />
                <h3 className="text-sm font-bold text-sky-500 uppercase tracking-widest">Transcrição Directa</h3>
              </div>
              
              <div className="text-2xl sm:text-3xl font-bold text-slate-800 break-words">
                {transcription && transcription.trim() !== "" ? (
                  <span className="text-emerald-600">{transcription.toUpperCase()}</span>
                ) : (
                  <span className="text-slate-400 font-medium text-xl">Comece a falar gestualmente...</span>
                )}
              </div>
            </div>
            
            <div className="bg-slate-800 rounded-3xl p-6 border border-slate-700">
              <p className="text-sm text-slate-300 font-medium leading-relaxed">
                🚀 <strong className="text-emerald-400">Modo C++ Hardware Acelerado:</strong> 
                <br/><br/>
                Para eliminar o lag do navegador, o túnel de conversão Base64 de JavaScript foi desativado. 
                A câmara é agora acedida pelo OpenCV no sistema com processamento neural (Mediapipe e Scikit) em threads de C++.
              </p>
            </div>
          </div>

        </main>
      </div>
    </motion.div>
  )
}
