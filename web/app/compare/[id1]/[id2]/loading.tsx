export default function CompareLoading() {
    return (
        <div className="max-w-[1280px] mx-auto px-6 md:px-12 py-8 animate-fadeIn">
            {/* Skeleton Breadcrumb */}
            <div className="h-4 w-48 bg-gray-200 rounded mb-8 animate-pulse" />

            {/* Skeleton Header */}
            <div className="text-center mb-12">
                <div className="h-10 w-64 bg-gray-200 rounded mx-auto mb-4 animate-pulse" />
                <div className="divider-accent mx-auto mb-6" />
                <div className="h-8 w-32 bg-gray-200 rounded mx-auto animate-pulse" />
            </div>

            {/* Skeleton Players */}
            <div className="grid md:grid-cols-2 gap-8 mb-12">
                {[...Array(2)].map((_, i) => (
                    <div key={i} className="card-elevated text-center">
                        <div className="w-16 h-16 bg-gray-200 rounded-full mx-auto mb-4 animate-pulse" />
                        <div className="h-8 w-48 bg-gray-200 rounded mx-auto mb-2 animate-pulse" />
                        <div className="h-4 w-32 bg-gray-200 rounded mx-auto mb-2 animate-pulse" />
                        <div className="flex justify-center gap-3">
                            <div className="h-5 w-16 bg-gray-200 rounded animate-pulse" />
                            <div className="h-5 w-16 bg-gray-200 rounded animate-pulse" />
                        </div>
                    </div>
                ))}
            </div>

            {/* Skeleton Radar Chart */}
            <div className="card-elevated mb-12">
                <div className="h-6 w-48 bg-gray-200 rounded mx-auto mb-6 animate-pulse" />
                <div className="h-[400px] w-full bg-gray-100 rounded-xl flex items-center justify-center">
                    <div className="spinner w-12 h-12" />
                </div>
            </div>

            {/* Skeleton Stats Table */}
            <div className="card-elevated">
                <div className="h-6 w-40 bg-gray-200 rounded mb-6 animate-pulse" />
                <div className="space-y-4">
                    {[...Array(6)].map((_, i) => (
                        <div key={i} className="flex justify-between items-center py-2 border-b border-gray-100">
                            <div className="h-4 w-32 bg-gray-200 rounded animate-pulse" />
                            <div className="h-4 w-16 bg-gray-200 rounded animate-pulse" />
                            <div className="h-4 w-16 bg-gray-200 rounded animate-pulse" />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
