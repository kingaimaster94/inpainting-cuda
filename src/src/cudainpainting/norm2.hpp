#ifndef __NORM2_HPP__
#define __NORM2_HPP__

template<bool B, class T = void> struct iftype {};
template<class T> struct iftype<true, T> { typedef T type; }; // enable_if

template<class T, T v> struct int_const { // integral_constant
    static const T value = v;
    typedef T value_type;
    typedef int_const type;
    operator value_type() const { return value; }
    value_type operator()() const { return value; }
};

typedef int_const<bool,true> ttype; // true_type
typedef int_const<bool,false> ftype; // false_type

template <class T, class U> struct same_as : ftype {};
template <class T> struct same_as<T, T> : ttype {};   // is_same


template <typename _Tp> struct is_norm2_type :
    int_const<bool, !same_as<_Tp,   int8_t>::value
                 && !same_as<_Tp,  uint8_t>::value
                 && !same_as<_Tp, uint16_t>::value
                 && !same_as<_Tp, uint32_t>::value>{};

template <typename _Tp, int cn> static inline typename iftype< is_norm2_type<_Tp>::value, _Tp >::
    type norm2(cv::Vec<_Tp, cn> a, cv::Vec<_Tp, cn> b) { return (a - b).dot(a - b); }

template <typename _Tp> static inline typename iftype< is_norm2_type<_Tp>::value, _Tp >::
    type norm2(const _Tp &a, const _Tp &b) { return (a - b)*(a - b); }

#endif /* __NORM2_HPP__ */
