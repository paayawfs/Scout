export default function Loading() {
    return (
        <div className="min-h-[60vh] flex items-center justify-center">
            <div className="flex flex-col items-center gap-4">
                <div className="spinner w-12 h-12" />
                <p className="text-gray-500 font-medium">Loading...</p>
            </div>
        </div>
    );
}
