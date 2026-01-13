export default function PlayerLoading() {
    return (
        <div className="max-w-[1280px] mx-auto px-4 sm:px-6 md:px-12 py-6 sm:py-8 animate-fadeIn">
            {/* Skeleton Breadcrumb */}
            <div className="h-4 w-32 bg-gray-200 rounded mb-8 animate-pulse" />

            {/* Skeleton Header */}
            <div className="mb-12">
                <div className="h-10 w-64 bg-gray-200 rounded mb-4 animate-pulse" />
                <div className="divider-accent mb-4" />
                <div className="flex gap-4">
                    <div className="h-6 w-32 bg-gray-200 rounded animate-pulse" />
                    <div className="h-6 w-20 bg-gray-200 rounded animate-pulse" />
                    <div className="h-6 w-16 bg-gray-200 rounded animate-pulse" />
                </div>
            </div>

            {/* Skeleton Stats */}
            <div className="mb-12">
                <div className="h-6 w-40 bg-gray-200 rounded mb-4 animate-pulse" />
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                    {[...Array(10)].map((_, i) => (
                        <div key={i} className="data-card py-4">
                            <div className="h-8 w-16 bg-gray-200 rounded mx-auto mb-2 animate-pulse" />
                            <div className="h-3 w-20 bg-gray-200 rounded mx-auto animate-pulse" />
                        </div>
                    ))}
                </div>
            </div>

            {/* Skeleton Similar Players */}
            <div>
                <div className="h-6 w-40 bg-gray-200 rounded mb-6 animate-pulse" />
                <div className="grid sm:grid-cols-2 gap-6">
                    {[...Array(4)].map((_, i) => (
                        <div key={i} className="card-elevated">
                            <div className="h-6 w-48 bg-gray-200 rounded mb-4 animate-pulse" />
                            <div className="h-4 w-32 bg-gray-200 rounded mb-3 animate-pulse" />
                            <div className="flex gap-2">
                                <div className="h-5 w-16 bg-gray-200 rounded animate-pulse" />
                                <div className="h-5 w-16 bg-gray-200 rounded animate-pulse" />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
