#ifndef KMEANS_LOGGING_H
#define KMEANS_LOGGING_H

#ifndef NDEBUG
#define DEBUG_PRINT(x) std::cout << x << std::endl
#define DEBUG_FLAG true
#else
#define DEBUG_PRINT(x) (void(0));
#define DEBUG_FLAG false
#endif

#define undefflag
#ifdef undefflag
#undef undefflag
#undef DEBUG_PRINT
#undef DEBUG_FLAG
#define DEBUG_PRINT(x) (void(0));
#define DEBUG_FLAG false
#endif


#endif