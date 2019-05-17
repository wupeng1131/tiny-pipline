#pragma once
#include <cstdint>
#include <cassert>
#include <numeric>

namespace vectorize {
	namespace detail {
	template<typename T>
	inline bool is_aligned(T, const typename T::value_type*) {
		return true;
	}

	template<typename T>

	inline bool is_aligned(T, const typename T::value_type* p1, const typename T::value_type* p2) {
		return is_aligned(T(), p1) && is_aligned(T(), p2);//? delete the ()
	}

	//traits
	template <typename T>
	struct generic{
		typedef T register_type;
		typedef T value_type;
		enum {
			unroll_size = 1
		};
		static register_type set1(const value_type& x) { return x; }
		static register_type zero() { return 0.0; }
		static register_type mul(const register_type& v1, const register_type& v2) { return v1 * v2; }
		static register_type add(const register_type& v1, const register_type& v2) { return v1 + v2; }
		static register_type load(const value_type* px) { return *px; }
		static register_type loadu(const value_type* px) { return *px; }
		static void store(value_type* px, const register_type& v) { *px = v; }
		static void storeu(value_type* px, const register_type& v) { *px = v; }
		static value_type resemble(const register_type& x) { return x; }

	};

	//generic dot-product
	template<typename T>
	inline typename T::value_type dot_product_nonaligned(const typename T::value_type* f1, const typename T::value_type* f2, unsigned int size) {
		typename T::register_type result = T::zero();

		for (unsigned int i = 0; i < size / T::unroll_size; i++) {
			result = T::add(result, T::mul( T::loadu(&f1[i*T::unroll_size]), T::loadu(&f2[i*T::unroll_size])));
		}
		typename T::value_type sum = T::resemble(result);

		for (unsigned int i = (size/T::unroll_size)*T::unroll_size; i < size; i++){
			sum += f1[i] * f2[i];
		}
		return sum;
	}

	template<typename T>
	inline typename T::value_type dot_product_aligned(const typename T::value_type* f1, const typename T::value_type* f2, unsigned int size) {
		typename T::register_type result = T::zero();

		assert(is_aligned(T(), f1));
		assert(is_aligned(T(), f2));

		for (unsigned int i = 0; i < size / T::unroll_size; i++) {
			result = T::add(result, T::mul(T::loadu(&f1[i*T::unroll_size]), T::loadu(&f2[i*T::unroll_size])));
		}
		typename T::value_type sum = T::resemble(result);

		for (unsigned int i = (size / T::unroll_size)*T::unroll_size; i < size; i++) {
			sum += f1[i] * f2[i];
		}
		return sum;
	}

	template<typename T>
	inline void muladd_aligned(const typename T::value_type* src, typename T::value_type c, unsigned int size, typename T::value_type* dst) {
		typename T::register_type factor = T::set1(c);

		for (unsigned int i = 0; i < size / T::unroll_size; i++) {
			typename T::register_type d = T::load(&dst[i * T::unroll_size]);
			typename T::register_type s = T::load(&src[i * T::unroll_size]);
			T::store(&dst[i*T::unroll_size], T::add(d, T::mul(s, factor)));
		}

		for (unsigned int i = (size / T::unroll_size)*T::unroll_size; i < size; i++) {
			dst[i] += src[i] * c;
		}
	}

	template<typename T>
	inline void muladd_nonaligned(const typename T::value_type* src, typename T::value_type c, unsigned int size, typename T::value_type* dst) {
		typename T::register_type factor = T::set1(c);

		for (unsigned int i = 0; i < size / T::unroll_size; i++) {
			typename T::register_type d = T::loadu(&dst[i*T::unroll_size]);
			typename T::register_type s = T::loadu(&src[i*T::unroll_size]);
			T::storeu(&dst[i*T::unroll_size], T::add(d, T::mul(s, factor)));
		}

		for (unsigned int i = (size / T::unroll_size)*T::unroll_size; i < size; i++)
			dst[i] += src[i] * c;
	}

	template<typename T>
	inline void reduce_nonaligned(const typename T::value_type* src, unsigned int size, typename T::value_type* dst) {
		for (unsigned int i = 0; i < size / T::unroll_size; i++) {
			typename T::register_type d = T::loadu(&dst[i*T::unroll_size]);
			typename T::register_type s = T::loadu(&src[i*T::unroll_size]);
			T::storeu(&dst[i*T::unroll_size], T::add(d, s));
		}

		for (unsigned int i = (size / T::unroll_size)*T::unroll_size; i < size; i++)
			dst[i] += src[i];
	}

	template<typename T>
	inline void reduce_aligned(const typename T::value_type* src, unsigned int size, typename T::value_type* dst) {
		for (unsigned int i = 0; i < size / T::unroll_size; i++) {
			typename T::register_type d = T::loadu(&dst[i*T::unroll_size]);
			typename T::register_type s = T::loadu(&src[i*T::unroll_size]);
			T::storeu(&dst[i*T::unroll_size], T::add(d, s));
		}

		for (unsigned int i = (size / T::unroll_size)*T::unroll_size; i < size; i++)
			dst[i] += src[i];
	}

	}//namespace detail

#define VECTORIZE_TYPE detail::generic<T>
	//dst[i] += c * src[i]
	template<typename T>
	void muladd(const T* src, T c, unsigned int size, T* dst) {
		if (detail::is_aligned(VECTORIZE_TYPE(), src, dst)) {
			detail::muladd_aligned<VECTORIZE_TYPE>(src, c, size, dst);
		}
		else {
			detail::muladd_nonaligned<VECTORIZE_TYPE>(src, c, size, dst);
		}
	}

	//sum(s1[i] * s2[i])
	template<typename T>
	T dot(const T*s1, const T*s2, unsigned int size) {
		if (detail::is_aligned(VECTORIZE_TYPE(), s1, s2))
			return detail::dot_product_aligned<VECTORIZE_TYPE>(s1, s2, size);
		else
			return detail::dot_product_nonaligned<VECTORIZE_TYPE>(s1, s2, size);
	}

	//dst[i] += src[i]
	template<typename T>
	void reduce(const T* src, unsigned int size, T* dst) {
		if (detail::is_aligned(VECTORIZE_TYPE(), src, dst)) {
			return detail::reduce_aligned<VECTORIZE_TYPE>(src, size, dst);
		}
		else {
			return detail::reduce_nonaligned<VECTORIZE_TYPE>(src, size, dst);
		}
	}
	}