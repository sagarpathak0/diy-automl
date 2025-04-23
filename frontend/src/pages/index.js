import { useState } from 'react'
import Head from 'next/head'
import Header from '../components/Header'
import Footer from '../components/Footer'
import FileUpload from '../components/FileUpload'
import ModelResult from '../components/ModelResult'
import LoadingState from '../components/LoadingState'
import { uploadFiles } from '../utils/api'

export default function Home() {
  const [trainingFile, setTrainingFile] = useState(null)
  const [predictionFile, setPredictionFile] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!trainingFile || !predictionFile) {
      setError('Both training and prediction files are required')
      return
    }

    setLoading(true)
    setError('')
    
    try {
      const result = await uploadFiles(trainingFile, predictionFile)
      setResults(result)
    } catch (err) {
      setError(err.message || 'An error occurred while processing your files')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      <Head>
        <title>DIY AutoML Platform</title>
        <meta name="description" content="Upload CSV files and automatically train ML models" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Header />

      <main className="flex-grow container mx-auto px-4 py-8">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-2">
              DIY AutoML Platform
            </h1>
            <p className="text-gray-600">
              Upload your data and let our system automatically select and train the best machine learning model
            </p>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            {!loading && !results && (
              <form onSubmit={handleSubmit}>
                <div className="space-y-6 text-black">
                  <FileUpload 
                    label="Training Data (CSV)" 
                    onChange={file => setTrainingFile(file)}
                    accept=".csv"
                    required
                  />
                  
                  <FileUpload 
                    label="Prediction Data (CSV)" 
                    onChange={file => setPredictionFile(file)}
                    accept=".csv"
                    required
                  />

                  {error && (
                    <div className="p-3 bg-red-100 text-red-700 rounded-md text-sm">
                      {error}
                    </div>
                  )}
                  
                  <button 
                    type="submit" 
                    className="btn btn-primary w-full cursor-pointer border-2 border-black py-3 rounded-md text-black font-semibold hover:bg-black hover:text-white transition duration-300 ease-in-out"
                    disabled={!trainingFile || !predictionFile}
                  >
                    Generate Predictions
                  </button>
                </div>
              </form>
            )}
            
            {loading && <LoadingState />}
            
            {results && <ModelResult results={results} />}

            {results && (
              <button 
                onClick={() => setResults(null)} 
                className="mt-6 btn btn-secondary px-3 bg-green-200 text-black cursor-pointer border-2 border-black py-3 rounded-md font-semibold hover:bg-black hover:text-white transition duration-300 ease-in-out"
              >
                Upload New Files
              </button>
            )}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}
