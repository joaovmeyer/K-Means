#pragma once

#include <x86intrin.h>

float squaredEuclideanDistance(const float* p1, const float* p2, int n) {
	float dst = 0.0f;
	for (size_t i = 0; i < n; ++i) {
		dst += (p1[i] - p2[i]) * (p1[i] - p2[i]);
	}

	return dst;
}


__m256* allocAVX(uint32_t n) {
	return new (std::align_val_t(32)) __m256[n];
}

void freeAVX(__m256* vecs) {
	::operator delete[] (vecs, std::align_val_t(32));
}


__m128 hminSSE(__m128 vec) {
	// vec = [x3, x2, x1, x0]
	vec = _mm_min_ps(vec, _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(2, 1, 0, 3))); // vec = [min(x3, x2), min(x2, x1), min(x1, x0), min(x0, x3)]
	vec = _mm_min_ps(vec, _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(1, 0, 3, 2)));
	// vec = [min(x3, x2, x1, x0), min(x2, x1, x0, x3), min(x1, x0, x3, x2), min(x0, x3, x2, x1)] (min of all elements)

	return vec;
}

__m256 hminAVX(__m256 vec) {

	vec = _mm256_min_ps(vec, _mm256_permute2f128_ps(vec, vec, 3)); // gets min of first 4 with last 4

	// same idea as SSE
	vec = _mm256_min_ps(vec, _mm256_permute_ps(vec, 0b01001110));
	vec = _mm256_min_ps(vec, _mm256_permute_ps(vec, 0b10110001));

	return vec;
}


int argminAVX(__m256 vec) {

	// example: vec = [5, 2, 3, 7, 11, 2, 4, 10]
	// indices:        8  6  5  4   3  2  1   0 --> argmin = 2

	__m256 minVal = hminAVX(vec); // minVal = [2, 2, 2, 2, 2, 2, 2, 2]
	int mask = _mm256_movemask_ps(_mm256_cmp_ps(minVal, vec, _CMP_EQ_OQ)); // mask = 0b01000100 (1 where vec's element == minVal)
	//                                                                                      ^
	return __builtin_ctz(mask); // index of first bit set in mask  -------------------------| = 2
}


// sets a single value of a __m256
template <int index>
void setValue(__m256& vec, float value) {
	constexpr uint8_t mask = 1 << index;
    vec = _mm256_blend_ps(vec, _mm256_set1_ps(value), mask); // value will be set where mask is 1
}

void setValue(__m256& vec, float value, uint32_t index = 0) {

	// index should be a compile-time constant
	switch (index % 8) { // modulo by power of two is basically free
		case 0: vec = _mm256_blend_ps(vec, _mm256_set1_ps(value), 1 << 0);
		case 1: vec = _mm256_blend_ps(vec, _mm256_set1_ps(value), 1 << 1);
		case 2: vec = _mm256_blend_ps(vec, _mm256_set1_ps(value), 1 << 2);
		case 3: vec = _mm256_blend_ps(vec, _mm256_set1_ps(value), 1 << 3);
		case 4: vec = _mm256_blend_ps(vec, _mm256_set1_ps(value), 1 << 4);
		case 5: vec = _mm256_blend_ps(vec, _mm256_set1_ps(value), 1 << 5);
		case 6: vec = _mm256_blend_ps(vec, _mm256_set1_ps(value), 1 << 6);
		case 7: vec = _mm256_blend_ps(vec, _mm256_set1_ps(value), 1 << 7);
	}
}


// calculates distances between one points to 8 points simultaneously
__m256 calculateDistances8x(const float* point, const __m256* points, int n) {

	// this works really well with low dimensions, opposed to calculating
	// one distance 8 dimensions at a time (for obvious reasons)
	__m256 dst = _mm256_setzero_ps();
	for (int d = 0; d < n; ++d) {
		__m256 diff = _mm256_sub_ps(_mm256_set1_ps(point[d]), points[d]);
		dst = _mm256_add_ps(dst, _mm256_mul_ps(diff, diff));
	}

	return dst;
}