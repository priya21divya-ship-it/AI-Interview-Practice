import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  RefreshCcw, Home, MessageSquare, Mic, Send, Star, Loader, X, Zap, Volume2, 
  Square, CornerUpLeft, MessageSquareQuote, Check, ArrowRight, Play, Upload, FileText, Image, Users
} from 'lucide-react';

// --- Global Configuration ---
const API_KEY = ""; 
const MODEL_NAME_TEXT = "gemini-2.5-flash-preview-09-2025";
const MODEL_NAME_TTS = "gemini-2.5-flash-preview-tts";

const ROLES = [
  'Software Engineer',
  'Sales Representative',
  'Retail Associate',
  'Data Scientist',
  'Product Manager',
  'Custom Role (via Job Description/Image)', // Added custom option
];

// Default question count is 5, as requested
const MAX_QUESTIONS = 5; 

// Voice configuration for the AI interviewer
const AI_VOICE = { prebuiltVoiceConfig: { voiceName: "Kore" } };

// Structured schema for the LLM to return the final feedback
const FEEDBACK_SCHEMA = {
  type: "OBJECT",
  properties: {
    score: { "type": "STRING", description: "Overall score out of 5, e.g., '3.5'. Must be a string." },
    summary: { "type": "STRING", description: "A concise summary of the candidate's performance." },
    breakdown: {
      type: "ARRAY",
      description: "Scores for key criteria.",
      items: {
        type: "OBJECT",
        properties: {
          label: { "type": "STRING", description: "Criteria name, e.g., 'Communication Clarity'." },
          rating: { "type": "NUMBER", description: "Score for this criterion out of 5." }
        },
        required: ["label", "rating"]
      }
    }
  },
  required: ["score", "summary", "breakdown"]
};


// --- Utility Functions (API and Audio) ---

// Base64 to ArrayBuffer for audio
const base64ToArrayBuffer = (base64) => {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
};

// PCM (Int16Array) to WAV Blob (required for playing the audio)
const pcmToWav = (pcmData, sampleRate) => {
    const numChannels = 1;
    const bitsPerSample = 16;

    const dataLength = pcmData.length * (bitsPerSample / 8);
    const buffer = new ArrayBuffer(44 + dataLength);
    const dataView = new DataView(buffer);

    // RIFF identifier
    writeString(dataView, 0, 'RIFF');
    // File size
    dataView.setUint32(4, 36 + dataLength, true);
    // Format
    writeString(dataView, 8, 'WAVE');
    // Format chunk identifier
    writeString(dataView, 12, 'fmt ');
    // Format chunk length
    dataView.setUint32(16, 16, true);
    // Sample format (1 for PCM)
    dataView.setUint16(20, 1, true);
    // Number of channels
    dataView.setUint16(22, numChannels, true);
    // Sample rate
    dataView.setUint32(24, sampleRate, true);
    // Byte rate (SampleRate * NumChannels * BitsPerSample/8)
    dataView.setUint32(28, sampleRate * numChannels * (bitsPerSample / 8), true);
    // Block align (NumChannels * BitsPerSample/8)
    dataView.setUint16(32, numChannels * (bitsPerSample / 8), true);
    // Bits per sample
    dataView.setUint16(34, bitsPerSample, true);
    // Data chunk identifier
    writeString(dataView, 36, 'data');
    // Data chunk length
    dataView.setUint32(40, dataLength, true);

    // Write PCM data
    let offset = 44;
    for (let i = 0; i < pcmData.length; i++) {
        dataView.setInt16(offset, pcmData[i], true);
        offset += 2;
    }

    return new Blob([dataView], { type: 'audio/wav' });
};

const writeString = (view, offset, string) => {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
};

// Exponential backoff retry logic for fetch
const retryFetch = async (url, options, retries = 3) => {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                // Log and throw for visibility, but don't stop silent retries
                console.warn(`Attempt ${i + 1} failed: HTTP error! status: ${response.status}`);
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response;
        } catch (error) {
            if (i < retries - 1) {
                const delay = Math.pow(2, i) * 1000;
                await new Promise(resolve => setTimeout(resolve, delay));
            } else {
                // Only throw the final error after max retries
                throw new Error("Gemini API request failed after multiple retries.");
            }
        }
    }
};

// Main Gemini API caller function for text generation
const callGeminiApi = async (contents, systemInstruction, generationConfig) => {
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL_NAME_TEXT}:generateContent?key=${API_KEY}`;

    const payload = {
        contents: [{ parts: contents }],
        systemInstruction: { parts: [{ text: systemInstruction }] },
        generationConfig: generationConfig || {},
    };

    try {
        const response = await retryFetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        
        // Handle structured JSON response
        if (generationConfig?.responseMimeType === "application/json") {
            const jsonText = result.candidates?.[0]?.content?.parts?.[0]?.text;
            if (jsonText) {
                try {
                    return jsonText; // Return the raw JSON string to be parsed later
                } catch (e) {
                    console.error("JSON parsing error:", e);
                    return "Error: Could not parse structured JSON content.";
                }
            }
        }
        
        // Handle plain text response
        return result.candidates?.[0]?.content?.parts?.[0]?.text || "Error: Could not generate content.";

    } catch (error) {
        console.error("Text API Error:", error);
        return `Error: Failed to fetch response (${error.message}).`;
    }
};

// Gemini API caller function for Text-to-Speech
const callGeminiTtsApi = async (text) => {
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL_NAME_TTS}:generateContent?key=${API_KEY}`;
    
    // Prompt the AI to say the question in an informative tone
    const ttsPrompt = `Say informatively: ${text}`;

    const payload = {
        contents: [{ parts: [{ text: ttsPrompt }] }],
        generationConfig: {
            responseModalities: ["AUDIO"],
            speechConfig: { voiceConfig: AI_VOICE }
        },
        model: MODEL_NAME_TTS
    };

    try {
        const response = await retryFetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        const part = result?.candidates?.[0]?.content?.parts?.[0];
        const audioData = part?.inlineData?.data;
        const mimeType = part?.inlineData?.mimeType;

        if (audioData && mimeType && mimeType.startsWith("audio/L16")) {
            const match = mimeType.match(/rate=(\d+)/);
            const sampleRate = match ? parseInt(match[1], 10) : 16000; // Default to 16000 if not found
            
            const pcmData = base64ToArrayBuffer(audioData);
            const pcm16 = new Int16Array(pcmData);
            const wavBlob = pcmToWav(pcm16, sampleRate);
            return URL.createObjectURL(wavBlob);
        }
        return null;
    } catch (error) {
        console.error("TTS API Error:", error);
        return null;
    }
};

// --- Utility Components ---

const StarRating = ({ rating, max = 5 }) => {
  const score = Math.round(parseFloat(rating));
  return (
    <div className="flex space-x-0.5">
      {[...Array(max)].map((_, i) => (
        <Star
          key={i}
          className={`w-5 h-5 transition-colors duration-200 ${
            i < score
              ? 'text-yellow-400 fill-yellow-400 drop-shadow-md'
              : 'text-gray-600 fill-gray-800'
          }`}
        />
      ))}
    </div>
  );
};

// --- Speech Recognition Hook ---

const useSpeechRecognition = (onResult) => {
    const [isListening, setIsListening] = useState(false);
    const recognitionRef = useRef(null);

    useEffect(() => {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.warn("Speech Recognition API is not supported in this browser.");
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
            const transcript = Array.from(event.results)
                .map(result => result[0])
                .map(result => result.transcript)
                .join('');
            onResult(transcript);
        };

        recognition.onend = () => {
            setIsListening(false);
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            setIsListening(false);
        };

        recognitionRef.current = recognition;

        return () => {
            if (recognitionRef.current) {
                recognitionRef.current.stop();
            }
        };
    }, [onResult]);

    const startListening = () => {
        if (recognitionRef.current) {
            recognitionRef.current.start();
            setIsListening(true);
        }
    };

    const stopListening = () => {
        if (recognitionRef.current) {
            recognitionRef.current.stop();
            setIsListening(false);
        }
    };

    return { isListening, startListening, stopListening };
};

// --- Screen Components ---

const RoleSelectionScreen = ({ onRoleSelect }) => {
  const [selectedRole, setSelectedRole] = useState(ROLES[0]);
  const [customContext, setCustomContext] = useState('');
  const [contextSource, setContextSource] = useState('none'); // 'none', 'text', 'image'
  const [isProcessingImage, setIsProcessingImage] = useState(false);
  const fileInputRef = useRef(null);
    
  const isCustomRole = selectedRole === 'Custom Role (via Job Description/Image)';

  // Helper for displaying custom alert without browser alert()
  const displayAlert = (message) => {
    const alertBox = document.createElement('div');
    alertBox.textContent = message;
    alertBox.style.cssText = `
        position: fixed; top: 20px; right: 20px; background-color: #dc2626; color: white; 
        padding: 10px 20px; border-radius: 8px; z-index: 9999; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: opacity 0.5s ease-in-out; font-family: sans-serif;
    `;
    document.body.appendChild(alertBox);
    setTimeout(() => {
        alertBox.style.opacity = '0';
        setTimeout(() => alertBox.remove(), 500);
    }, 3000);
  };


  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        displayAlert("Please select an image file."); 
        return;
      }

      setIsProcessingImage(true);
      setCustomContext('');
      setContextSource('image');

      try {
        const base64Data = await fileToBase64(file);
        
        // Use LLM to analyze the image content
        const extractedText = await callGeminiImageApi(base64Data, file.type);
        setCustomContext(extractedText);
      } catch (error) {
        console.error("Image processing error:", error);
        displayAlert("Failed to process image. Please try again."); 
        setCustomContext('');
        setContextSource('none');
      } finally {
        setIsProcessingImage(false);
        // Clear file input value to allow selecting the same file again
        event.target.value = null; 
      }
    }
  };

  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.onerror = error => reject(error);
      reader.readAsDataURL(file);
    });
  };

  const callGeminiImageApi = async (base64ImageData, mimeType) => {
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL_NAME_TEXT}:generateContent?key=${API_KEY}`;
    const prompt = "Analyze the provided image of a job posting or description. Extract all relevant details: the job title, required skills, main responsibilities, and company information. Present the extracted information as a concise, structured text block suitable for an AI interviewer to base their questions on. If no job description is visible, state that clearly."

    const payload = {
        contents: [
            {
                role: "user",
                parts: [
                    { text: prompt },
                    {
                        inlineData: {
                            mimeType: mimeType,
                            data: base64ImageData
                        }
                    }
                ]
            }
        ],
    };

    try {
        const response = await retryFetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        return result.candidates?.[0]?.content?.parts?.[0]?.text || "Error: Could not extract text from image.";
    } catch (error) {
        console.error("Image Analysis API Error:", error);
        return `Error: Failed to analyze image content (${error.message}).`;
    }
  };


  return (
    <div className="p-8 md:p-10 bg-gray-800/90 backdrop-blur-md rounded-2xl shadow-2xl shadow-indigo-900/50 w-full max-w-2xl border border-indigo-700/50 transition-all duration-500">
      <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400 mb-2 text-center">
        AI Interview Coach
      </h1>
      <p className="text-gray-300 text-center mb-8 text-md">
        Select a role or provide a job description/image to start your practice.
      </p>

      {/* Role Selection Grid */}
      <label className="block text-gray-400 mb-4 text-sm font-semibold uppercase tracking-wider">
        SELECT INTERVIEW ROLE
      </label>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-8">
        {ROLES.map((role) => {
            const isSelected = selectedRole === role;
            const isCustom = role.includes('Custom Role');
            const roleName = role.replace(' (via Job Description/Image)', '');
            
            let Icon = Users;
            if (roleName === 'Software Engineer') Icon = FileText;
            if (roleName === 'Data Scientist') Icon = Star;
            if (isCustom) Icon = Zap;

            return (
                <button
                    key={role}
                    onClick={() => {
                        setSelectedRole(role);
                        if (!isCustom) {
                            setCustomContext('');
                            setContextSource('none');
                        }
                    }}
                    className={`p-4 rounded-xl text-center font-semibold transition-all duration-300 transform border-2 flex flex-col items-center justify-center min-h-[100px] ${
                        isSelected
                            ? 'bg-indigo-600 border-indigo-500 text-white shadow-lg shadow-indigo-600/50 scale-[1.03]'
                            : 'bg-gray-700/50 border-gray-700 text-gray-300 hover:bg-gray-600/70 hover:border-indigo-500'
                    }`}
                >
                    <Icon className={`w-6 h-6 mx-auto mb-1 ${isCustom ? 'text-yellow-400' : ''}`} />
                    <span className='mt-2 text-sm'>{roleName}</span>
                </button>
            );
        })}
      </div>


      {/* Custom Context Input */}
      {isCustomRole && (
        <div className="mb-8 p-4 bg-gray-700/50 rounded-xl border border-indigo-600/50 shadow-inner">
            <h2 className='text-xl font-bold text-indigo-400 mb-4 flex items-center'>
                <FileText className='w-5 h-5 mr-2'/> ✨ Provide Custom Context
            </h2>
            <p className='text-sm text-gray-400 mb-3'>Paste a Job Description or upload an image to generate ultra-specific questions.</p>

            <div className='flex space-x-3 mb-3'>
                {/* Text Input Button */}
                <button 
                    onClick={() => {
                        setContextSource('text'); 
                        if(customContext === '' || customContext === 'Paste your job description here...') setCustomContext('Paste your job description here...');
                    }}
                    className={`flex-1 py-2 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center ${
                        contextSource === 'text' ? 'bg-indigo-600 text-white shadow-md' : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
                    }`}
                >
                    <FileText className='w-4 h-4 mr-2'/> Paste Text
                </button>

                {/* Image Upload Button */}
                <button 
                    onClick={() => fileInputRef.current.click()}
                    disabled={isProcessingImage}
                    className={`flex-1 py-2 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center ${
                        isProcessingImage ? 'bg-gray-700 text-gray-500 cursor-wait' : 
                        contextSource === 'image' ? 'bg-indigo-600 text-white shadow-md' : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
                    }`}
                >
                    {isProcessingImage ? (
                        <Loader className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                        <Image className='w-4 h-4 mr-2'/>
                    )}
                    Upload Image
                </button>
                <input
                    type="file"
                    accept="image/*"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                />
            </div>
            
            {contextSource !== 'none' && (
                <div className="mt-4">
                    <textarea
                        value={customContext}
                        onChange={(e) => {
                            setCustomContext(e.target.value);
                            setContextSource('text'); // Treat any manual edit as text input
                        }}
                        placeholder="Paste your job description here..."
                        rows={6}
                        disabled={isProcessingImage}
                        className="w-full p-3 rounded-lg bg-gray-800 text-gray-200 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 border border-gray-600 resize-none disabled:opacity-70"
                    />
                    {isProcessingImage && (
                        <p className='text-sm text-indigo-400 mt-2 flex items-center'>
                            <Loader className="w-4 h-4 mr-2 animate-spin" />
                            Analyzing image with Gemini...
                        </p>
                    )}
                    {contextSource === 'image' && !isProcessingImage && (
                        <p className='text-sm text-green-400 mt-2 flex items-center'>
                            <Check className='w-4 h-4 mr-1'/> Text extracted successfully. Review and edit if needed.
                        </p>
                    )}
                </div>
            )}
        </div>
      )}


      <button
        onClick={() => {
            if (isCustomRole && !customContext.trim()) {
                displayAlert("Please provide a job description or image for a custom interview."); 
                return;
            }
            onRoleSelect(selectedRole, customContext);
        }}
        disabled={isCustomRole && (!customContext.trim() || isProcessingImage)}
        className={`w-full py-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-bold text-lg rounded-xl transition-all duration-300 shadow-xl shadow-indigo-600/40 transform hover:scale-[1.01] ${
            isCustomRole && (!customContext.trim() || isProcessingImage) ? 'opacity-50 cursor-not-allowed' : ''
        }`}
      >
        Start Mock Interview
      </button>
    </div>
  );
};

// Component to display the Refinement Modal
const RefinementModal = ({ data, onUseRefined, onKeepOriginal }) => {
    if (!data) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm transition-opacity duration-300">
            <div className="bg-gray-900 rounded-2xl max-w-3xl w-full p-8 shadow-2xl shadow-yellow-900/50 border border-yellow-700/50 transform scale-100 transition-transform duration-300">
                <h2 className="text-3xl font-bold text-yellow-400 mb-6 flex items-center">
                    <MessageSquareQuote className='w-7 h-7 mr-2'/> Refine Your Answer
                </h2>
                
                <div className="space-y-6 max-h-[70vh] overflow-y-auto custom-scrollbar pr-3">
                    <div className="p-4 bg-gray-800 rounded-xl border border-gray-700 shadow-inner">
                        <h3 className="text-xl font-semibold text-gray-200 mb-2 border-b border-gray-700 pb-1">AI Critique:</h3>
                        <p className="text-sm text-gray-400 italic leading-relaxed markdown-content" dangerouslySetInnerHTML={{ __html: data.feedback.replace(/\n/g, '<br/>') }} />
                    </div>

                    <div className="p-5 bg-gradient-to-r from-indigo-900/50 to-purple-900/50 rounded-xl border border-indigo-700 shadow-xl">
                        <h3 className="text-xl font-semibold text-indigo-300 mb-2 flex items-center">
                            <ArrowRight className='w-5 h-5 mr-2 text-yellow-400'/> Refined Answer:
                        </h3>
                        <p className="text-gray-100 leading-relaxed markdown-content" dangerouslySetInnerHTML={{ __html: data.refinedAnswer.replace(/\n/g, '<br/>') }} />
                    </div>
                    
                    <div className="p-4 bg-gray-700/50 rounded-xl border border-gray-600">
                        <h3 className="text-xl font-semibold text-gray-200 mb-2 border-b border-gray-600 pb-1">Your Original Answer:</h3>
                        <p className="text-sm text-gray-300 leading-relaxed markdown-content" dangerouslySetInnerHTML={{ __html: data.originalAnswer.replace(/\n/g, '<br/>') }} />
                    </div>
                </div>

                <div className="flex justify-end space-x-4 mt-8 border-t border-gray-700 pt-6">
                    <button
                        onClick={onKeepOriginal}
                        className="py-3 px-6 bg-gray-700 hover:bg-gray-600 text-white rounded-xl transition-all duration-300 flex items-center font-medium shadow-md hover:shadow-lg"
                    >
                        Keep Original
                    </button>
                    <button
                        onClick={onUseRefined}
                        className="py-3 px-6 bg-yellow-600 hover:bg-yellow-500 text-gray-900 font-bold rounded-xl transition-all duration-300 flex items-center shadow-lg shadow-yellow-600/40 transform hover:scale-[1.03]"
                    >
                        <CornerUpLeft className='w-5 h-5 mr-2'/> Use Refined Answer
                    </button>
                </div>
            </div>
        </div>
    );
};

// Component to display the Final Feedback
const FeedbackScreen = ({ feedback, history, role, onRestart }) => {
    // Ensure score is a number for StarRating, handle potential errors
    const overallScore = parseFloat(feedback.score) || 0;

    return (
      <div className="p-8 md:p-10 bg-gray-800/90 backdrop-blur-md rounded-2xl shadow-2xl shadow-indigo-900/50 w-full max-w-4xl border border-indigo-700/50 transition-all duration-500">
        <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-teal-400 mb-2 text-center flex items-center justify-center">
            <Star className="w-8 h-8 mr-3 text-yellow-400 fill-yellow-400"/> Interview Assessment
        </h1>
        <p className="text-gray-300 text-center mb-8 text-lg">
          Role: <span className="font-semibold text-indigo-400">{role}</span>
        </p>

        {/* Overall Score Card */}
        <div className="bg-gray-700/50 p-6 rounded-xl mb-8 border border-green-700/50 shadow-inner">
            <h2 className="text-2xl font-bold text-green-400 mb-4">Overall Performance</h2>
            <div className="flex items-center space-x-4">
                <div className="text-5xl font-extrabold text-white">
                    {feedback.score}
                </div>
                <div className="flex flex-col">
                    <StarRating rating={overallScore} />
                    <span className="text-gray-400 text-sm">out of 5.0</span>
                </div>
            </div>
            <p className="mt-4 text-gray-200 italic leading-relaxed">{feedback.summary}</p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
            
            {/* Score Breakdown */}
            <div className="bg-gray-700 p-6 rounded-xl border border-indigo-700 shadow-md">
                <h2 className="text-xl font-bold text-indigo-300 mb-4 flex items-center">
                    <Zap className="w-5 h-5 mr-2"/> Detailed Metrics
                </h2>
                <ul className="space-y-4">
                    {feedback.breakdown?.map((item, index) => (
                        <li key={index} className="border-b border-gray-600 pb-2">
                            <div className="font-medium text-gray-200">{item.label}</div>
                            <div className="flex items-center justify-between text-sm mt-1">
                                <StarRating rating={item.rating} />
                                <span className="font-bold text-indigo-400">{(item.rating || 0).toFixed(1)} / 5.0</span>
                            </div>
                        </li>
                    ))}
                </ul>
            </div>

            {/* Conversation History */}
            <div className="bg-gray-700 p-6 rounded-xl border border-indigo-700 shadow-md">
                <h2 className="text-xl font-bold text-indigo-300 mb-4 flex items-center">
                    <MessageSquare className="w-5 h-5 mr-2"/> Conversation Log
                </h2>
                <div className="space-y-4 max-h-72 overflow-y-auto custom-scrollbar pr-2">
                    {history.map((message, index) => (
                        <div key={index} className={`p-3 rounded-lg ${message.role === 'ai' ? 'bg-indigo-900/50 text-indigo-200' : 'bg-gray-600/50 text-gray-200'}`}>
                            <div className={`font-semibold text-sm mb-1 ${message.role === 'ai' ? 'text-indigo-400' : 'text-green-400'}`}>
                                {message.role === 'ai' ? 'Interviewer' : 'Your Answer'}
                            </div>
                            <p className="text-sm">{message.text}</p>
                        </div>
                    ))}
                </div>
            </div>
        </div>

        <button
            onClick={onRestart}
            className="w-full mt-8 py-4 bg-gradient-to-r from-green-600 to-teal-600 hover:from-green-700 hover:to-teal-700 text-white font-bold text-lg rounded-xl transition-all duration-300 shadow-xl shadow-green-600/40 transform hover:scale-[1.01] flex items-center justify-center"
        >
            <RefreshCcw className="w-5 h-5 mr-2"/> Start New Interview
        </button>
      </div>
    );
};


// The main Interview Component
const InterviewScreen = ({ role, customContext, onComplete }) => {
  const [responseText, setResponseText] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefining, setIsRefining] = useState(false);
  const [audioStates, setAudioStates] = useState({}); // { messageIndex: 'loading' | 'ready' | 'playing' }
  const [refinementData, setRefinementData] = useState(null); 
  
  const chatEndRef = useRef(null);
  const audioRef = useRef(new Audio());
  const recognitionAvailable = 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
  
  const completedQuestions = chatHistory.filter(m => m.role === 'user').length;
  const currentQuestionNumber = completedQuestions + 1;
  const totalQuestions = MAX_QUESTIONS;
  const progressPercent = (completedQuestions / totalQuestions) * 100;


  const onSpeechResult = useCallback((transcript) => {
    setResponseText(transcript);
  }, []);
  
  const { isListening, startListening, stopListening } = useSpeechRecognition(onSpeechResult);

  // Scroll to the latest message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  // Audio cleanup on unmount
  useEffect(() => {
    return () => {
      audioRef.current.pause();
      if (audioRef.current.src) {
        URL.revokeObjectURL(audioRef.current.src);
      }
    };
  }, []);

  const handleAudioEnd = useCallback(() => {
    const playingIndex = Object.keys(audioStates).find(key => audioStates[key] === 'playing');
    if (playingIndex) {
        setAudioStates(prev => ({ ...prev, [playingIndex]: 'ready' }));
    }
  }, [audioStates]);

  const playAudio = useCallback(async (text, index) => {
    // Stop any currently playing audio
    if (!audioRef.current.paused) {
        audioRef.current.pause();
        handleAudioEnd();
    }
    
    // Set status to loading
    setAudioStates(prev => ({ ...prev, [index]: 'loading' }));

    const audioUrl = await callGeminiTtsApi(text);
    
    if (audioUrl) {
      audioRef.current.src = audioUrl;
      audioRef.current.onended = handleAudioEnd;
      audioRef.current.onerror = () => {
          console.error("Audio playback error");
          setAudioStates(prev => ({ ...prev, [index]: 'ready' }));
      };
      audioRef.current.play().then(() => {
        setAudioStates(prev => ({ ...prev, [index]: 'playing' }));
      }).catch(e => {
        console.error("Audio playback failed:", e);
        setAudioStates(prev => ({ ...prev, [index]: 'ready' }));
      });
    } else {
      setAudioStates(prev => ({ ...prev, [index]: 'ready' }));
    }
  }, [handleAudioEnd]);

  const generateNextQuestion = useCallback(async (history) => {
    setIsLoading(true);

    const contextInstruction = customContext 
        ? `The candidate is applying for a role defined by the following context: "${customContext}". Base your questions SPECIFICALLY on the skills and responsibilities mentioned in this context, starting with the job title itself if present.`
        : `The candidate is interviewing for a standard ${role} role.`;

    const systemPrompt = `You are a professional AI interviewer for a ${role} position. ${contextInstruction} Your task is to generate the next interview question. You must use the STAR method for at least one behavioral question across the ${MAX_QUESTIONS} questions. Do not generate more than one question at a time. The current question number is ${history.filter(m => m.role === 'ai').length + 1} out of ${MAX_QUESTIONS}.`;

    const contents = history.map(msg => ({ text: msg.text }));

    const question = await callGeminiApi(
      [{ text: `Based on the conversation so far, generate a concise, single, highly relevant interview question.` }],
      systemPrompt
    );
    
    // 1. Add question to history immediately
    const newIndex = history.length;
    setChatHistory(prev => [...prev, { role: 'ai', text: question }]);
    
    // 2. Set input available
    setIsLoading(false);
    
    // 3. Start audio generation non-blocking in the background
    setAudioStates(prev => ({ ...prev, [newIndex]: 'loading' }));
    
    const audioUrl = await callGeminiTtsApi(question);
    
    if (audioUrl) {
        // 4. Once URL is ready, play it and mark as ready/playing
        audioRef.current.src = audioUrl;
        audioRef.current.onended = handleAudioEnd;
        audioRef.current.play().then(() => {
            setAudioStates(prev => ({ ...prev, [newIndex]: 'playing' }));
        }).catch(e => {
            console.error("Auto playback failed on ready:", e);
            setAudioStates(prev => ({ ...prev, [newIndex]: 'ready' }));
        });
    } else {
        setAudioStates(prev => ({ ...prev, [newIndex]: 'ready' }));
    }

  }, [role, customContext, handleAudioEnd]);

  // Initial Question on mount
  useEffect(() => {
    if (chatHistory.length === 0) {
      generateNextQuestion([]);
    }
  }, [generateNextQuestion, chatHistory.length]);

  const handleStopAudio = () => {
    if (!audioRef.current.paused) {
        audioRef.current.pause();
        handleAudioEnd();
    }
  };
  
  const handleToggleListening = () => {
    if (!recognitionAvailable) {
        // Helper for displaying custom alert without browser alert()
        const displayAlert = (message) => {
            const alertBox = document.createElement('div');
            alertBox.textContent = message;
            alertBox.style.cssText = `
                position: fixed; top: 20px; right: 20px; background-color: #dc2626; color: white; 
                padding: 10px 20px; border-radius: 8px; z-index: 9999; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                transition: opacity 0.5s ease-in-out; font-family: sans-serif;
            `;
            document.body.appendChild(alertBox);
            setTimeout(() => {
                alertBox.style.opacity = '0';
                setTimeout(() => alertBox.remove(), 500);
            }, 3000);
          };
        displayAlert("Speech recognition is not supported in your browser."); 
        return;
    }
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  // The existing Refine Answer feature (Gemini Text API)
  const handleRefineAnswer = async () => {
    if (!responseText.trim() || isLoading || isRefining) return;

    setIsRefining(true);
    setRefinementData(null); // Clear previous refinement data

    // Stop listening if it was active
    if (isListening) stopListening();

    const refinementPrompt = `Analyze the following answer provided by a candidate for the role of ${role}, specifically considering the interview context: "${customContext}". Provide constructive feedback on how to improve the answer's clarity, structure, and content, focusing on the STAR method if applicable. Then, rewrite the answer to be highly effective, concise, and professional. Use the following format STRICTLY:\n\n**FEEDBACK:** [Your critique]\n\n**REFINED ANSWER:** [Your suggested polished answer]`;

    try {
        const critique = await callGeminiApi(
            [{ text: `Candidate's current answer: "${responseText.trim()}"` }],
            refinementPrompt
        );
        
        // Parse the structured critique
        const feedbackMatch = critique.match(/\*\*FEEDBACK:\*\*([\s\S]*?)(?=\n\n\*\*REFINED ANSWER:\*\*|$)/i);
        const refinedAnswerMatch = critique.match(/\*\*REFINED ANSWER:\*\*([\s\S]*)/i);

        const feedbackText = feedbackMatch ? feedbackMatch[1].trim() : "No specific feedback found. The AI might have had trouble parsing the structure.";
        const refinedAnswerText = refinedAnswerMatch ? refinedAnswerMatch[1].trim() : responseText.trim();
        
        setRefinementData({
            feedback: feedbackText,
            refinedAnswer: refinedAnswerText,
            originalAnswer: responseText.trim()
        });

    } catch (error) {
        console.error("Refinement API Error:", error);
        // Fallback for user experience
        setRefinementData({
            feedback: "Error: Failed to fetch refinement from AI.",
            refinedAnswer: responseText.trim(),
            originalAnswer: responseText.trim()
        });
    } finally {
        setIsRefining(false);
    }
  };
  
  const handleUseRefined = () => {
    if (refinementData) {
        setResponseText(refinementData.refinedAnswer);
        setRefinementData(null);
    }
  };
  
  const handleSendAnswer = async (answer) => {
    if (!answer.trim() || isLoading || isRefining) return;

    handleStopAudio(); // Ensure audio stops when sending an answer
    setIsLoading(true);
    setResponseText('');
    setRefinementData(null); // Clear refinement data after sending an answer

    // 1. Add user message to history
    const newHistory = [...chatHistory, { role: 'user', text: answer.trim() }];
    setChatHistory(newHistory);

    // 2. Check if interview is over
    if (newHistory.filter(m => m.role === 'user').length >= totalQuestions) {
        await handleEndInterview(newHistory);
        return;
    }

    // 3. If not over, generate next question
    await generateNextQuestion(newHistory);
  };
  
  const handleEndInterview = async (finalHistory) => {
    setIsLoading(true);
    
    // Concatenate chat history for the final prompt
    const conversationSummary = finalHistory.map(m => `${m.role === 'ai' ? 'Interviewer' : 'Candidate'}: ${m.text}`).join('\n');
    
    const systemPrompt = `You are a professional Interview Assessor. Based on the following conversation for a candidate applying for the role of ${role} (Context: "${customContext}"), provide a structured performance report. You MUST follow the JSON schema EXACTLY. The overall score should be reflective of the candidate's performance across all questions, focusing on clarity, technical accuracy, relevance, and ability to use structured response methods (like STAR).`;

    const finalPrompt = `Provide the final assessment and score using the provided JSON structure, based on this interview:\n\n---\n${conversationSummary}\n---`;
    
    try {
        const resultJson = await callGeminiApi(
            [{ text: finalPrompt }],
            systemPrompt,
            {
                responseMimeType: "application/json",
                responseSchema: FEEDBACK_SCHEMA,
            }
        );
        
        const parsedFeedback = JSON.parse(resultJson);
        onComplete(parsedFeedback, finalHistory, role);
    } catch (error) {
        console.error("Final Feedback API Error:", error);
        // Fallback to a simple error state
        onComplete({
            score: 'N/A', 
            summary: `Failed to generate feedback: ${error.message}. Please check the console for details.`, 
            breakdown: []
        }, finalHistory, role);
    } finally {
        setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[90vh] w-full max-w-4xl bg-gray-800/90 backdrop-blur-md rounded-2xl shadow-2xl shadow-indigo-900/50 border border-indigo-700/50 transition-all duration-500">
        
        {/* Header */}
        <div className="p-4 border-b border-indigo-700 flex justify-between items-center rounded-t-2xl bg-gray-700/50">
            <h2 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400 flex items-center">
                <MessageSquare className='w-5 h-5 mr-2'/> Mock Interview: {role}
            </h2>
            <div className='flex items-center space-x-3'>
                <span className={`text-sm font-semibold ${completedQuestions >= totalQuestions ? 'text-green-400' : 'text-indigo-400'}`}>
                    Q {Math.min(currentQuestionNumber, totalQuestions)} / {totalQuestions}
                </span>
                {completedQuestions < totalQuestions && (
                    <button 
                        onClick={() => handleEndInterview(chatHistory)}
                        className='px-3 py-1 bg-yellow-600 hover:bg-yellow-500 text-gray-900 font-medium text-sm rounded-lg transition-colors duration-200'
                        disabled={isLoading}
                    >
                        End Interview
                    </button>
                )}
            </div>
        </div>
        
        {/* Progress Bar */}
        <div className="w-full bg-gray-700 h-2">
            <div 
                className={`h-full bg-gradient-to-r ${completedQuestions >= totalQuestions ? 'from-green-400 to-teal-400' : 'from-indigo-400 to-purple-400'} transition-all duration-500 rounded-r-lg`} 
                style={{ width: `${progressPercent}%` }}
            />
        </div>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
            {chatHistory.map((message, index) => (
                <div 
                    key={index} 
                    className={`flex ${message.role === 'ai' ? 'justify-start' : 'justify-end'}`}
                >
                    <div className={`max-w-[80%] p-4 rounded-xl shadow-lg relative ${
                        message.role === 'ai' 
                        ? 'bg-indigo-900/50 text-indigo-100 border border-indigo-700/50' 
                        : 'bg-gray-700 text-gray-100 border border-gray-600'
                    }`}>
                        <div className={`font-semibold text-sm mb-1 ${message.role === 'ai' ? 'text-indigo-300' : 'text-green-300'}`}>
                            {message.role === 'ai' ? 'Interviewer' : 'You'}
                        </div>
                        <p className='text-sm leading-relaxed whitespace-pre-wrap'>{message.text}</p>
                        
                        {/* Audio Button for AI Messages */}
                        {message.role === 'ai' && (
                            <button
                                onClick={() => {
                                    if (audioStates[index] === 'playing') {
                                        handleStopAudio();
                                    } else {
                                        playAudio(message.text, index);
                                    }
                                }}
                                disabled={audioStates[index] === 'loading' || isLoading}
                                className={`absolute -right-10 top-1 p-2 rounded-full transition-colors duration-200 ${
                                    audioStates[index] === 'playing' ? 'bg-red-500 text-white' : 'bg-indigo-600 hover:bg-indigo-500 text-white'
                                } disabled:opacity-50`}
                                title={audioStates[index] === 'playing' ? 'Stop Speaking' : 'Listen'}
                            >
                                {audioStates[index] === 'loading' && <Loader className="w-4 h-4 animate-spin"/>}
                                {audioStates[index] === 'ready' && <Volume2 className="w-4 h-4"/>}
                                {audioStates[index] === 'playing' && <Square className="w-4 h-4"/>}
                                {!(audioStates[index]) && <Play className="w-4 h-4"/>}
                            </button>
                        )}
                    </div>
                </div>
            ))}
            {isLoading && completedQuestions < totalQuestions && (
                <div className="flex justify-start">
                    <div className="max-w-[80%] p-4 rounded-xl bg-indigo-900/50 text-indigo-100 border border-indigo-700/50">
                        <Loader className="w-5 h-5 animate-spin mr-2 inline" />
                        <span className='text-sm'>AI is preparing the next question...</span>
                    </div>
                </div>
            )}
            {isLoading && completedQuestions >= totalQuestions && (
                <div className="flex justify-start">
                    <div className="max-w-[80%] p-4 rounded-xl bg-green-900/50 text-green-100 border border-green-700/50">
                        <Loader className="w-5 h-5 animate-spin mr-2 inline" />
                        <span className='text-sm'>Generating final performance assessment...</span>
                    </div>
                </div>
            )}
            <div ref={chatEndRef} />
        </div>
        
        {/* Input Area */}
        {completedQuestions < totalQuestions && (
            <div className="p-4 border-t border-indigo-700 bg-gray-700/50 rounded-b-2xl">
                <div className="flex space-x-3">
                    <textarea
                        value={responseText}
                        onChange={(e) => setResponseText(e.target.value)}
                        placeholder={isListening ? "Listening... Speak your answer now." : "Type or speak your interview answer here..."}
                        rows={2}
                        disabled={isLoading || isRefining}
                        className="flex-1 p-3 rounded-xl bg-gray-800 text-gray-200 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 border border-gray-600 resize-none disabled:opacity-70"
                    />

                    <div className="flex flex-col space-y-2">
                        {/* Refine Button */}
                        <button
                            onClick={handleRefineAnswer}
                            disabled={!responseText.trim() || isLoading || isRefining || isListening}
                            className={`flex items-center justify-center p-3 rounded-xl font-semibold text-sm transition-all duration-300 ${
                                refinementData 
                                ? 'bg-yellow-600 hover:bg-yellow-500 text-gray-900 shadow-md shadow-yellow-600/40' 
                                : 'bg-gray-600 hover:bg-gray-500 text-white'
                            } disabled:opacity-50`}
                            title="Get AI feedback & refine your current answer"
                        >
                            {isRefining ? (
                                <Loader className="w-5 h-5 animate-spin" />
                            ) : (
                                <Star className="w-5 h-5" />
                            )}
                        </button>
                        
                        {/* Mic Button */}
                        <button
                            onClick={handleToggleListening}
                            disabled={isLoading || isRefining || !recognitionAvailable}
                            className={`flex items-center justify-center p-3 rounded-xl font-semibold text-sm transition-all duration-300 ${
                                isListening 
                                ? 'bg-red-600 hover:bg-red-500 text-white shadow-md shadow-red-600/40' 
                                : 'bg-indigo-600 hover:bg-indigo-500 text-white'
                            } disabled:opacity-50`}
                            title={isListening ? "Stop Recording" : "Start Voice Input"}
                        >
                            <Mic className="w-5 h-5" />
                        </button>
                    </div>

                    {/* Send Button */}
                    <button
                        onClick={() => handleSendAnswer(responseText)}
                        disabled={!responseText.trim() || isLoading || isRefining}
                        className="bg-green-600 hover:bg-green-500 text-white p-3 rounded-xl font-bold transition-all duration-300 shadow-xl shadow-green-600/40 transform hover:scale-[1.05] disabled:opacity-50"
                        title="Submit Answer"
                    >
                        <Send className="w-6 h-6" />
                    </button>
                </div>
            </div>
        )}

        {/* Completion Message */}
        {completedQuestions >= totalQuestions && !isLoading && (
             <div className="p-4 border-t border-indigo-700 bg-gray-700/50 rounded-b-2xl text-center">
                 <p className='text-lg font-semibold text-green-400'>
                     Interview completed! Scroll up to see the assessment being generated.
                 </p>
             </div>
        )}

        {/* Refinement Modal */}
        {refinementData && (
            <RefinementModal 
                data={refinementData}
                onUseRefined={handleUseRefined}
                onKeepOriginal={() => setRefinementData(null)}
            />
        )}
    </div>
  );
};


// Main Application Component
const App = () => {
  const [screen, setScreen] = useState('Home'); // 'Home', 'Interview', 'Feedback'
  const [role, setRole] = useState(null);
  const [customContext, setCustomContext] = useState('');
  const [feedbackData, setFeedbackData] = useState(null);
  const [interviewHistory, setInterviewHistory] = useState([]);

  const handleRoleSelect = (selectedRole, context) => {
    setRole(selectedRole);
    setCustomContext(context);
    setScreen('Interview');
  };

  const handleInterviewComplete = (feedback, history, finalRole) => {
    setFeedbackData(feedback);
    setInterviewHistory(history);
    setRole(finalRole);
    setScreen('Feedback');
  };

  const handleRestart = () => {
    setRole(null);
    setCustomContext('');
    setFeedbackData(null);
    setInterviewHistory([]);
    setScreen('Home');
  };
  
  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4" style={{ fontFamily: 'Inter, sans-serif' }}>
      {screen === 'Home' && <RoleSelectionScreen onRoleSelect={handleRoleSelect} />}
      {screen === 'Interview' && role && (
        <InterviewScreen
          role={role}
          customContext={customContext}
          onComplete={handleInterviewComplete}
        />
      )}
      {screen === 'Feedback' && feedbackData && (
        <FeedbackScreen
          feedback={feedbackData}
          history={interviewHistory}
          role={role}
          onRestart={handleRestart}
        />
      )}
    </div>
  );
};

export default App;
