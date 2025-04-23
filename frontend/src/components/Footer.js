export default function Footer() {
  return (
    <footer className="bg-white">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-center">
          <p className="text-sm text-gray-500">
            © {new Date().getFullYear()} DIY AutoML Platform. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}