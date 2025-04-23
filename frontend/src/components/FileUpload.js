import { useState } from 'react'

export default function FileUpload({ label, onChange, accept, required = false }) {
  const [fileName, setFileName] = useState('')
  
  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setFileName(file.name)
      onChange(file)
    } else {
      setFileName('')
      onChange(null)
    }
  }

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-dark">
        {label} {required && <span className="text-red-500">*</span>}
      </label>
      
      <div className="flex items-center">
        <label className="relative cursor-pointer bg-white rounded-md border border-gray-300 p-2 hover:bg-gray-50 flex-grow">
          <span className="flex items-center justify-center">
            <svg 
              className="h-6 w-6 text-gray-400" 
              stroke="currentColor" 
              fill="none" 
              viewBox="0 0 24 24"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" 
              />
            </svg>
            <span className="ml-3 text-sm text-dark">
              {fileName ? fileName : 'Choose file...'}
            </span>
          </span>
          <input 
            type="file" 
            className="sr-only" 
            accept={accept}
            onChange={handleFileChange}
            required={required}
          />
        </label>
      </div>
      
      <p className="text-xs text-gray-500">
        Only CSV files are supported
      </p>
    </div>
  )
}