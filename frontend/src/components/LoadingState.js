export default function LoadingState() {
  return (
    <div className="text-center py-12">
      <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
      <p className="mt-4 text-black">
        Processing your data...
      </p>
      <p className="text-sm text-black text-medium mt-2">
        This may take a few moments depending on the size of your dataset
      </p>
    </div>
  )
}