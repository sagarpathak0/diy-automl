import axios from 'axios'

const API_BASE_URL = 'http://localhost:5000/api'

export const uploadFiles = async (trainingFile, predictionFile) => {
  const formData = new FormData()
  formData.append('training_file', trainingFile)
  formData.append('prediction_file', predictionFile)

  try {
    const response = await axios.post(`${API_BASE_URL}/automl`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    return response.data
  } catch (error) {
    throw new Error(
      error.response?.data?.error || 
      'Failed to process your files. Please try again.'
    )
  }
}

export const downloadPredictions = async (url) => {
  try {
    // Fix URL construction to avoid duplicate /api/ in the path
    let downloadUrl;
    
    if (url.startsWith('http')) {
      // If it's already a full URL, use it as is
      downloadUrl = url;
    } else if (url.startsWith('/api/')) {
      // If it starts with /api/, add only the base hostname
      downloadUrl = `http://localhost:5000${url}`;
    } else {
      // Otherwise append to API_BASE_URL
      downloadUrl = `${API_BASE_URL}${url}`;
    }
    
    console.log('Downloading from URL:', downloadUrl);
    
    const response = await axios.get(downloadUrl, {
      responseType: 'blob'
    })
    
    // Create a blob from the response data
    const blob = new Blob([response.data], { type: 'text/csv' })
    
    // Create a temporary URL for the blob
    const blobUrl = window.URL.createObjectURL(blob)
    
    // Create a temporary link element
    const link = document.createElement('a')
    link.href = blobUrl
    link.setAttribute('download', 'predictions.csv')
    
    // Append the link to the body
    document.body.appendChild(link)
    
    // Click the link to start the download
    link.click()
    
    // Clean up
    setTimeout(() => {
      document.body.removeChild(link)
      window.URL.revokeObjectURL(blobUrl)
    }, 100)
    
    return true
  } catch (error) {
    console.error('Download error:', error)
    throw error;  // Re-throw the error so it can be handled by the caller
  }
}

export const downloadModel = async (url) => {
  try {
    // Fix URL construction to avoid duplicate /api/ in the path
    let downloadUrl;
    
    if (url.startsWith('http')) {
      // If it's already a full URL, use it as is
      downloadUrl = url;
    } else if (url.startsWith('/api/')) {
      // If it starts with /api/, add only the base hostname
      downloadUrl = `http://localhost:5000${url}`;
    } else {
      // Otherwise append to API_BASE_URL
      downloadUrl = `${API_BASE_URL}${url}`;
    }
    
    console.log('Downloading model from URL:', downloadUrl);
    
    const response = await axios.get(downloadUrl, {
      responseType: 'blob'
    })
    
    // Create a blob from the response data
    const blob = new Blob([response.data], { type: 'application/zip' })
    
    // Create a temporary URL for the blob
    const blobUrl = window.URL.createObjectURL(blob)
    
    // Create a temporary link element
    const link = document.createElement('a')
    link.href = blobUrl
    link.setAttribute('download', 'diy_automl_model.zip')
    
    // Append the link to the body
    document.body.appendChild(link)
    
    // Click the link to start the download
    link.click()
    
    // Clean up
    setTimeout(() => {
      document.body.removeChild(link)
      window.URL.revokeObjectURL(blobUrl)
    }, 100)
    
    return true
  } catch (error) {
    console.error('Model download error:', error)
    throw error;
  }
}