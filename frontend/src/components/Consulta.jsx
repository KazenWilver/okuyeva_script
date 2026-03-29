import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Video, Activity } from 'lucide-react'
import { useState, useEffect } from 'react'

export default function Consulta() {
  const navigate = useNavigate()
  
  const [transcription, setTranscription] = useState("")
  const [isConnected, setIsConnected] = useState(false)
  const [gestureType, setGestureType] = useState("none")
  const [confidence, setConfidence] = useState(0)
  const [lstmReady, setLstmReady] = useState(false)
  
  useEffect(() => {
    const fetchState = async () => {
      try {
        const res = await fetch("http://localhost:8000/tracking_state")
        const data = await res.json()
        setTranscription(data.gesture || "")
        setIsConnected(data.camera_active)
        setGestureType(data.gesture_type || "none")
        setConfidence(Math.round((data.confidence || 0) * 100))
        setLstmReady(data.lstm_available || false)
      } catch { setIsConnected(false) }
    }
    const id = setInterval(fetchState, 100)
    return () => clearInterval(id)
  }, [])

  return (
    <motion.div 
      className="w-full h-full flex flex-col px-4 sm:px-6 py-4 sm:py-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4 flex-shrink-0">
        <div className="flex items-center gap-3">
          <button 
            onClick={() => navigate('/menu')}
            className="strata-card w-9 h-9 sm:w-10 sm:h-10 flex items-center justify-center cursor-pointer !rounded-full"
          >
            <ArrowLeft size={15} />
          </button>
          <span className="mono-tag">Em Consulta</span>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-400 pulse-ring' : 'bg-gray-300'}`} />
          <span className="mono-label">{isConnected ? 'ACTIVO' : 'A LIGAR...'}</span>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-[1.3fr_0.7fr] gap-4 flex-1 min-h-0">
        
        {/* Video Feed */}
        <div className="strata-card flex items-center justify-center min-h-[200px] sm:min-h-[280px] !overflow-hidden">
          <img 
            src="http://localhost:8000/video_feed" 
            alt="Video Feed" 
            className="w-full h-full object-cover absolute inset-0" 
            onError={(e) => { e.target.style.display = 'none' }}
            onLoad={(e) => { e.target.style.display = 'block' }}
            style={{ display: 'none', borderRadius: '12px' }}
          />
          {!isConnected && (
            <div className="flex flex-col items-center gap-3">
              <Video size={28} strokeWidth={1} style={{ color: 'rgba(0,0,0,0.15)' }} />
              <span className="mono-label">A AGUARDAR API...</span>
            </div>
          )}
        </div>

        {/* Right Panel */}
        <div className="flex flex-col gap-3 sm:gap-4">
          
          {/* Transcrição */}
          <div className="strata-card p-5 sm:p-7 flex-1 flex flex-col !hover:transform-none" style={{ minHeight: '100px' }}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Activity size={12} className="text-emerald-400" />
                <span className="mono-label">Transcrição</span>
              </div>
              {gestureType !== "none" && (
                <span className="mono-label px-2.5 py-1 rounded-full" style={{ background: 'rgba(52,211,153,0.1)', color: 'var(--accent-green)', fontSize: '0.6rem' }}>
                  {gestureType === "dynamic" ? "LSTM" : "STATIC"}
                </span>
              )}
            </div>
            
            <div className="flex-1 flex flex-col justify-center">
              {transcription && transcription.trim() !== "" ? (
                <div>
                  <div className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tighter leading-none break-words uppercase">
                    {transcription}
                  </div>
                  <span className="mono-label mt-3 block">CONFIANÇA: {confidence}%</span>
                </div>
              ) : (
                <p className="text-sm" style={{ color: 'var(--text-mono)' }}>
                  Comece a falar gestualmente...
                </p>
              )}
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-3 sm:gap-4">
            <div className="strata-card p-4 sm:p-5">
              <span className="mono-label block mb-2">Confiança</span>
              <div className="flex items-end gap-1">
                <span className="text-2xl sm:text-3xl font-extrabold leading-none">{confidence}</span>
                <span className="mono-label">%</span>
              </div>
              <div className="mt-3 h-[3px] w-full rounded-full" style={{ background: 'rgba(0,0,0,0.06)' }}>
                <motion.div 
                  className="h-full rounded-full bg-emerald-400"
                  initial={{ width: '0%' }}
                  animate={{ width: `${confidence}%` }}
                  transition={{ duration: 0.4 }}
                />
              </div>
            </div>

            <div className="strata-card p-4 sm:p-5">
              <span className="mono-label block mb-2">Motor</span>
              <div className="mono-label leading-[1.9]" style={{ fontSize: '0.6rem' }}>
                MEDIAPIPE HOLISTIC<br/>
                {lstmReady ? 'LSTM BiDIR ✓' : 'AGUARDANDO TREINO'}<br/>
                &lt;50MS LATÊNCIA
              </div>
              <div className="flex items-center gap-2 mt-2">
                <div className={`w-1.5 h-1.5 rounded-full ${lstmReady ? 'bg-emerald-400' : 'bg-gray-300'}`} />
                <span className="mono-label" style={{ fontSize: '0.55rem' }}>
                  {lstmReady ? 'ACTIVO' : 'ESPERA'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
