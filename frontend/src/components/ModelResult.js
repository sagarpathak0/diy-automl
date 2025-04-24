import { useState } from 'react'
import { downloadPredictions, downloadModel } from '../utils/api'

export default function ModelResult({ results }) {
  const [activeTab, setActiveTab] = useState('overview')
  const [downloading, setDownloading] = useState(false)
  const [downloadingModel, setDownloadingModel] = useState(false)
  
  const handleDownload = async () => {
    setDownloading(true)
    try {
      await downloadPredictions(results.download_url)
    } catch (error) {
      alert('Failed to download predictions: ' + error.message)
    } finally {
      setDownloading(false)
    }
  }
  
  const handleModelDownload = async () => {
    setDownloadingModel(true)
    try {
      await downloadModel(results.model_download_url)
    } catch (error) {
      alert('Failed to download model: ' + error.message)
    } finally {
      setDownloadingModel(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-green-100 p-4 rounded-md">
        <h3 className="font-semibold text-green-800">Model Training Complete!</h3>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="flex -mb-px">
          <button
            onClick={() => setActiveTab('overview')}
            className={`mr-8 py-4 px-1 ${
              activeTab === 'overview'
                ? 'border-b-2 border-primary font-medium text-primary text-blue-500'
                : 'text-gray-500 hover:text-gray-700 border-b-2 border-transparent'
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => setActiveTab('features')}
            className={`mr-8 py-4 px-1 ${
              activeTab === 'features'
                ? 'border-b-2 border-primary font-medium text-primary text-blue-500'
                : 'text-gray-500 hover:text-gray-700 border-b-2 border-transparent'
            }`}
          >
            Feature Importance
          </button>
          <button
            onClick={() => setActiveTab('metrics')}
            className={`py-4 px-1 ${
              activeTab === 'metrics'
                ? 'border-b-2 border-primary font-medium text-primary text-blue-500'
                : 'text-gray-500 hover:text-gray-700 border-b-2 border-transparent'
            }`}
          >
            Metrics
          </button>
        </nav>
      </div>

      {/* Tab Content */}
      <div className="space-y-6">
        {activeTab === 'overview' && (
          <div className="space-y-4">
            <div className="flex justify-between bg-black p-4 rounded-md">
              <span className="font-medium">Selected Model:</span>
              <span className="text-primary font-semibold">{results.model_type}</span>
            </div>
            
            <p className="text-sm text-black">
              Based on the characteristics of your data, we've selected the best model for your use case.
            </p>
            
            <div className="mt-6 flex flex-col sm:flex-row gap-3">
              <button
                onClick={handleDownload}
                disabled={downloading}
                className="btn btn-primary inline-flex items-center font-bold justify-center text-black hover:bg-green-600 hover:text-white px-2 py-3 rounded border-2 cursor-pointer border-black"
              >
                {downloading ? (
                  <span className="inline-flex items-center">
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Downloading...
                  </span>
                ) : (
                  <>
                    <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                    Download Predictions
                  </>
                )}
              </button>
              
              <button
                onClick={handleModelDownload}
                disabled={downloadingModel}
                className="btn btn-primary inline-flex items-center font-bold justify-center text-black hover:bg-green-600 hover:text-white px-2 py-3 rounded border-2 cursor-pointer border-black"
              >
                {downloadingModel ? (
                  <span className="inline-flex items-center">
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Downloading...
                  </span>
                ) : (
                  <>
                    <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M10 2a4 4 0 00-4 4v2H5a2 2 0 00-2 2v8a2 2 0 002 2h10a2 2 0 002-2v-8a2 2 0 00-2-2h-1V6a4 4 0 00-4-4zm3 8H7v-2a3 3 0 016 0v2z" clipRule="evenodd" />
                    </svg>
                    Download Model
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {activeTab === 'features' && (
          <div className="space-y-4">
            <h3 className="font-bold text-2xl text-black">Feature Importance</h3>
            {results.feature_importance && Object.keys(results.feature_importance).length > 0 ? (
              <div className="space-y-3 text-black font-bold">
                {Object.entries(results.feature_importance)
                  .sort((a, b) => b[1] - a[1])
                  .map(([feature, importance]) => (
                    <div key={feature} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span>{feature}</span>
                        <span className="font-medium">{(importance * 100).toFixed(2)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-primary h-2 rounded-full"
                          style={{ width: `${importance * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500">
                Feature importance not available for this model type.
              </p>
            )}
          </div>
        )}

        {activeTab === 'metrics' && (
          <div className="space-y-4">
            <h3 className="font-bold text-gray-700">Model Performance Metrics</h3>
            {results.metrics && Object.keys(results.metrics).length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-green-600">
                {Object.entries(results.metrics).map(([metric, value]) => (
                  <div key={metric} className="bg-gray-50 p-4 rounded-md">
                    <div className="text-xs text-gray-500 uppercase">{metric}</div>
                    <div className="text-xl font-semibold">{typeof value === 'number' ? value.toFixed(4) : value}</div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500">
                No metrics available for this model type.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}