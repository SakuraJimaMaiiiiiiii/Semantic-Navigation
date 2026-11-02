#pragma once
#ifdef _WIN32
#include <thread>
#include <mutex>
#include <chrono>
#include <cerrno>
using pthread_mutex_t = std::mutex;
struct pthread_t { std::thread worker; explicit operator bool() const {return worker.joinable();} };
inline int pthread_mutex_init(std::mutex*, const void*) {return 0;}
inline int pthread_mutex_destroy(std::mutex*) {return 0;}
inline int pthread_mutex_lock(std::mutex* m) {m->lock();return 0;}
inline int pthread_mutex_unlock(std::mutex* m) {m->unlock();return 0;}
inline int pthread_mutex_trylock(std::mutex* m) {return m->try_lock()?0:EBUSY;}
inline int pthread_create(pthread_t* t,const void*,void* (*fn)(void*),void* arg) {t->worker=std::thread([=]{fn(arg);});return 0;}
inline int pthread_join(pthread_t& t,void**) {if(t.worker.joinable()) t.worker.join();return 0;}
inline void usleep(unsigned long us) {std::this_thread::sleep_for(std::chrono::microseconds(us));}
#else
#include <pthread.h>
#include <unistd.h>
#endif
