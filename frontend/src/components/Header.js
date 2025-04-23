export default function Header() {
  return (
    <header className="bg-white shadow">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <svg 
              className="h-8 w-8 text-primary" 
              fill="currentColor" 
              viewBox="0 0 24 24"
            >
              <path d="M12 16c1.626 0 3.06-.797 3.943-2.02M12 16v4M12 20h3M15 16h2a2 2 0 002-2v-1M19 13V9a2 2 0 00-2-2h-2M15 7h-3M12 7V3M12 3H9M9 7H7a2 2 0 00-2 2v1M5 10v4a2 2 0 002 2h2M9 16c-.937 0-1.813-.407-2.412-1.114" 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth="2"
              />
            </svg>
            <h1 className="ml-2 text-xl font-bold text-gray-800">DIY AutoML</h1>
          </div>
          
          <div>
            <a 
              href="https://github.com/sagarpathak0/diy-automl" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-gray-600 hover:text-gray-900"
            >
              <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24">
                <path
                  fillRule="evenodd"
                  d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                  clipRule="evenodd"
                />
              </svg>
            </a>
          </div>
        </div>
      </div>
    </header>
  )
}