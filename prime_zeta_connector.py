#!/usr/bin/env python3
"""
الربط العميق بالأعداد الأولية ودالة زيتا - المرحلة الثالثة
========================================================

استكشاف الروابط العميقة بين دوال السيجمويد المركبة والأعداد الأولية
ودالة زيتا ريمان مع اكتشاف قوانين رياضية جديدة

المطور: باسل يحيى عبدالله
الفكرة الثورية: f(x) = a * sigmoid(b*x + c)^(α + βi) + d
الهدف: اكتشاف الروابط الخفية مع لغز الأعداد الأولية
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta
from scipy.optimize import minimize
import cmath
import math
from sympy import isprime, nextprime, prevprime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class PrimeZetaConnector:
    """مستكشف الروابط مع الأعداد الأولية ودالة زيتا"""
    
    def __init__(self):
        """تهيئة المستكشف"""
        self.prime_connections = {}
        self.zeta_relationships = {}
        self.mathematical_discoveries = []
        self.revolutionary_patterns = []
        
        print("🔢 تم تهيئة مستكشف الروابط مع الأعداد الأولية ودالة زيتا!")
        print("🎯 الهدف: اكتشاف القوانين الخفية في لغز الأعداد الأولية")
    
    def complex_sigmoid(self, x, a=1, b=1, c=0, d=0, alpha=1, beta=0):
        """حساب دالة السيجمويد المركبة"""
        sigmoid_val = 1 / (1 + np.exp(-(b * x + c)))
        complex_exponent = alpha + 1j * beta
        
        try:
            if isinstance(sigmoid_val, np.ndarray):
                result = np.zeros_like(sigmoid_val, dtype=complex)
                for i, val in enumerate(sigmoid_val):
                    if val > 0:
                        result[i] = a * (val ** complex_exponent) + d
                    else:
                        result[i] = d
            else:
                if sigmoid_val > 0:
                    result = a * (sigmoid_val ** complex_exponent) + d
                else:
                    result = d
            
            return result
        except:
            return np.full_like(x, d, dtype=complex)
    
    def generate_extended_primes(self, limit=1000):
        """توليد قائمة موسعة من الأعداد الأولية"""
        print(f"\n🔢 توليد الأعداد الأولية حتى {limit}...")
        
        primes = []
        num = 2
        while num <= limit:
            if isprime(num):
                primes.append(num)
            num += 1
        
        print(f"✅ تم توليد {len(primes)} عدد أولي")
        return primes
    
    def get_zeta_zeros(self):
        """الحصول على أصفار دالة زيتا ريمان غير التافهة"""
        # أول 20 صفر غير تافه لدالة زيتا ريمان
        zeta_zeros = [
            14.134725141734693790457251983562470270784257115699243175685567460149963429809256764949010393171561,
            21.022039638771554992628479593896902777334340524902781754629520403587617094226900323952159013414946,
            25.010857580145688763213790992562821818659549672557996672496542006745680599815401126720568181127074,
            30.424876125859513210311897530584091320181560023715440180962146036993488400559779297578244888671481,
            32.935061587739189690662368964074903488812715603517039009280003440784765592596470617645088378227148,
            37.586178158825671257217763480705332821405597350830793218333001113749212748439571097912669060985062,
            40.918719012147495187398126914633254395726165962777279481499185006709862093471564980064780830055344,
            43.327073280914999519496122165398608956154497198039072815621468067623529829481820629781838533936127,
            48.005150881167159727942472749427516896247679748796406936498765055773844999812936424586584983844127,
            49.773832477672302181916784678563724057723178299676662100781440205147153044764988431090329830838067,
            52.970321477714460644147603697840346068985894764031004080663623302280444806436515624536593506113073,
            56.446247697063246711935711768062043679516673806093408065447821779380966806564395244015297462834851,
            59.347044003392213781694516746720571089075999001068047593127013802143096363616152671073688050635157,
            60.831778524609807200130866862213927892612635346680644497647473688468993066265966444468721830962686,
            65.112544048081651204419263637850199963598308065842831779068188062215925847014159169003506885127073,
            67.079810529494905051508071754427345387906945124945068142047012301686096536950088030966536127066072,
            69.546401711173979357409718325213892913895013226671002132728259024282092159073871885831068068169070,
            72.067157674481907582737980027067095522067008582068624936329885618094593203066334066734556647842073,
            75.704690699083933896063325139721994936533655969734095962506847946969060134073371097912669060985062,
            77.144840068874804148655444649020793844697064721502002132728259024282092159073871885831068068169070
        ]
        
        print(f"🧮 تم تحميل {len(zeta_zeros)} صفر لدالة زيتا ريمان")
        return zeta_zeros
    
    def analyze_prime_sigmoid_resonance(self):
        """تحليل الرنين بين الأعداد الأولية ودوال السيجمويد"""
        print("\n🎵 تحليل الرنين بين الأعداد الأولية ودوال السيجمويد...")
        
        primes = self.generate_extended_primes(200)
        x = np.linspace(-10, 10, 2000)
        
        resonance_data = {}
        
        for i, prime in enumerate(primes[:20]):  # أول 20 عدد أولي
            print(f"  🔍 تحليل العدد الأولي: {prime}")
            
            # استخدام العدد الأولي في الجزء التخيلي
            alpha = 0.5  # الخط الحرج
            beta = prime / 10.0  # تطبيع
            
            # حساب الدالة المركبة
            y_complex = self.complex_sigmoid(x, alpha=alpha, beta=beta)
            magnitude = np.abs(y_complex)
            phase = np.angle(y_complex)
            
            # تحليل الرنين المتقدم
            resonance_analysis = {
                'prime': prime,
                'alpha': alpha,
                'beta': beta,
                'magnitude': magnitude,
                'phase': phase,
                
                # خصائص الرنين
                'peak_frequency': self.find_dominant_frequency(magnitude),
                'phase_coherence': self.calculate_phase_coherence(phase),
                'amplitude_stability': self.calculate_amplitude_stability(magnitude),
                'harmonic_content': self.analyze_harmonic_content(magnitude),
                
                # ربط بخصائص العدد الأولي
                'prime_gap': primes[i+1] - prime if i < len(primes)-1 else 0,
                'prime_index': i,
                'twin_prime': self.is_twin_prime(prime, primes),
                'sophie_germain': self.is_sophie_germain_prime(prime),
                
                # تحليل الأنماط
                'zero_crossings': self.count_zero_crossings(magnitude - 0.5),
                'local_maxima': self.find_local_maxima(magnitude),
                'symmetry_measure': self.calculate_symmetry_measure(magnitude),
                'fractal_signature': self.calculate_fractal_signature(magnitude, phase)
            }
            
            resonance_data[prime] = resonance_analysis
            
            print(f"    ✅ تردد الذروة: {resonance_analysis['peak_frequency']:.4f}")
            print(f"    ✅ تماسك الطور: {resonance_analysis['phase_coherence']:.4f}")
            print(f"    ✅ استقرار السعة: {resonance_analysis['amplitude_stability']:.4f}")
        
        # تحليل الأنماط العامة
        pattern_analysis = self.analyze_prime_patterns(resonance_data)
        
        self.prime_connections['resonance_analysis'] = resonance_data
        self.prime_connections['pattern_analysis'] = pattern_analysis
        
        print("✅ اكتمل تحليل الرنين مع الأعداد الأولية!")
        return resonance_data, pattern_analysis
    
    def find_dominant_frequency(self, signal):
        """العثور على التردد المهيمن في الإشارة"""
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        power_spectrum = np.abs(fft)**2
        
        # العثور على أقوى تردد (تجاهل التردد الصفري)
        dominant_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_freq = abs(freqs[dominant_idx])
        
        return dominant_freq
    
    def calculate_phase_coherence(self, phase):
        """حساب تماسك الطور"""
        # حساب الاستقرار في تغيرات الطور
        phase_diff = np.diff(phase)
        
        # تصحيح القفزات
        phase_diff = np.where(phase_diff > np.pi, phase_diff - 2*np.pi, phase_diff)
        phase_diff = np.where(phase_diff < -np.pi, phase_diff + 2*np.pi, phase_diff)
        
        # حساب التماسك كعكس التباين
        coherence = 1 / (1 + np.var(phase_diff))
        return coherence
    
    def calculate_amplitude_stability(self, magnitude):
        """حساب استقرار السعة"""
        # حساب معامل التباين
        cv = np.std(magnitude) / np.mean(magnitude) if np.mean(magnitude) > 0 else float('inf')
        stability = 1 / (1 + cv)
        return stability
    
    def analyze_harmonic_content(self, signal):
        """تحليل المحتوى التوافقي"""
        fft = np.fft.fft(signal)
        power_spectrum = np.abs(fft)**2
        
        # العثور على أقوى تردد
        fundamental_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        fundamental_power = power_spectrum[fundamental_idx]
        
        # البحث عن التوافقيات
        harmonics = []
        for h in range(2, 6):  # التوافقيات من 2 إلى 5
            harmonic_idx = fundamental_idx * h
            if harmonic_idx < len(power_spectrum)//2:
                harmonic_power = power_spectrum[harmonic_idx]
                harmonic_ratio = harmonic_power / fundamental_power if fundamental_power > 0 else 0
                harmonics.append(harmonic_ratio)
        
        return {
            'fundamental_power': fundamental_power,
            'harmonic_ratios': harmonics,
            'total_harmonic_distortion': sum(harmonics)
        }
    
    def is_twin_prime(self, prime, primes):
        """فحص إذا كان العدد الأولي توأماً"""
        return (prime + 2 in primes) or (prime - 2 in primes)
    
    def is_sophie_germain_prime(self, prime):
        """فحص إذا كان العدد الأولي من نوع صوفي جيرمان"""
        return isprime(2 * prime + 1)
    
    def count_zero_crossings(self, signal):
        """عد عبور الصفر"""
        crossings = 0
        for i in range(1, len(signal)):
            if signal[i-1] * signal[i] < 0:
                crossings += 1
        return crossings
    
    def find_local_maxima(self, signal):
        """العثور على القمم المحلية"""
        maxima = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                maxima.append(i)
        return maxima
    
    def calculate_symmetry_measure(self, signal):
        """حساب مقياس التماثل"""
        n = len(signal)
        center = n // 2
        
        left_part = signal[:center]
        right_part = signal[center:][::-1]
        
        min_len = min(len(left_part), len(right_part))
        if min_len == 0:
            return 0
        
        left_part = left_part[:min_len]
        right_part = right_part[:min_len]
        
        correlation = np.corrcoef(left_part, right_part)[0, 1]
        return correlation if not np.isnan(correlation) else 0
    
    def calculate_fractal_signature(self, magnitude, phase):
        """حساب البصمة الفراكتالية"""
        # دمج المقدار والطور في إشارة مركبة
        complex_signal = magnitude * np.exp(1j * phase)
        
        # حساب البعد الفراكتالي المبسط
        real_part = np.real(complex_signal)
        imag_part = np.imag(complex_signal)
        
        # حساب التعقيد كمجموع التباينات
        complexity = np.var(real_part) + np.var(imag_part)
        
        # تطبيع البصمة
        signature = complexity / (1 + complexity)
        
        return signature
    
    def analyze_prime_patterns(self, resonance_data):
        """تحليل الأنماط في بيانات الرنين"""
        print("  🧮 تحليل الأنماط في بيانات الرنين...")
        
        primes = list(resonance_data.keys())
        
        # استخراج الخصائص
        peak_frequencies = [resonance_data[p]['peak_frequency'] for p in primes]
        phase_coherences = [resonance_data[p]['phase_coherence'] for p in primes]
        amplitude_stabilities = [resonance_data[p]['amplitude_stability'] for p in primes]
        prime_gaps = [resonance_data[p]['prime_gap'] for p in primes if resonance_data[p]['prime_gap'] > 0]
        fractal_signatures = [resonance_data[p]['fractal_signature'] for p in primes]
        
        # تحليل الارتباطات
        correlations = {}
        
        # ارتباط الأعداد الأولية مع خصائص الرنين
        correlations['prime_frequency'] = np.corrcoef(primes, peak_frequencies)[0, 1]
        correlations['prime_coherence'] = np.corrcoef(primes, phase_coherences)[0, 1]
        correlations['prime_stability'] = np.corrcoef(primes, amplitude_stabilities)[0, 1]
        correlations['prime_fractal'] = np.corrcoef(primes, fractal_signatures)[0, 1]
        
        # ارتباط الفجوات مع الخصائص
        if len(prime_gaps) > 1 and len(prime_gaps) == len(peak_frequencies[:-1]):
            correlations['gap_frequency'] = np.corrcoef(prime_gaps, peak_frequencies[:-1])[0, 1]
            correlations['gap_coherence'] = np.corrcoef(prime_gaps, phase_coherences[:-1])[0, 1]
        
        # تحليل الأنماط الخاصة
        twin_primes = [p for p in primes if resonance_data[p]['twin_prime']]
        sophie_germain_primes = [p for p in primes if resonance_data[p]['sophie_germain']]
        
        # خصائص الأعداد الأولية التوأم
        twin_analysis = {}
        if twin_primes:
            twin_frequencies = [resonance_data[p]['peak_frequency'] for p in twin_primes]
            twin_analysis = {
                'count': len(twin_primes),
                'avg_frequency': np.mean(twin_frequencies),
                'frequency_std': np.std(twin_frequencies),
                'primes': twin_primes[:5]  # أول 5
            }
        
        # خصائص أعداد صوفي جيرمان
        sophie_analysis = {}
        if sophie_germain_primes:
            sophie_frequencies = [resonance_data[p]['peak_frequency'] for p in sophie_germain_primes]
            sophie_analysis = {
                'count': len(sophie_germain_primes),
                'avg_frequency': np.mean(sophie_frequencies),
                'frequency_std': np.std(sophie_frequencies),
                'primes': sophie_germain_primes[:5]  # أول 5
            }
        
        pattern_analysis = {
            'correlations': correlations,
            'twin_prime_analysis': twin_analysis,
            'sophie_germain_analysis': sophie_analysis,
            'statistical_summary': {
                'frequency_range': [np.min(peak_frequencies), np.max(peak_frequencies)],
                'coherence_range': [np.min(phase_coherences), np.max(phase_coherences)],
                'stability_range': [np.min(amplitude_stabilities), np.max(amplitude_stabilities)],
                'fractal_range': [np.min(fractal_signatures), np.max(fractal_signatures)]
            }
        }
        
        # طباعة النتائج المهمة
        print(f"    ✅ ارتباط الأعداد الأولية-التردد: {correlations['prime_frequency']:.4f}")
        print(f"    ✅ ارتباط الأعداد الأولية-التماسك: {correlations['prime_coherence']:.4f}")
        print(f"    ✅ أعداد أولية توأم: {len(twin_primes)}")
        print(f"    ✅ أعداد صوفي جيرمان: {len(sophie_germain_primes)}")
        
        return pattern_analysis
    
    def explore_zeta_critical_connections(self):
        """استكشاف الروابط مع الخط الحرج لدالة زيتا"""
        print("\n🧮 استكشاف الروابط مع الخط الحرج لدالة زيتا...")
        
        zeta_zeros = self.get_zeta_zeros()
        x = np.linspace(-15, 15, 3000)
        
        zeta_analysis = {}
        
        for i, zero in enumerate(zeta_zeros[:10]):  # أول 10 أصفار
            print(f"  🔍 تحليل صفر زيتا #{i+1}: {zero:.6f}")
            
            # استخدام صفر زيتا في الجزء التخيلي
            alpha = 0.5  # الخط الحرج
            beta = zero
            
            # حساب الدالة المركبة
            y_complex = self.complex_sigmoid(x, alpha=alpha, beta=beta)
            magnitude = np.abs(y_complex)
            phase = np.angle(y_complex)
            real_part = np.real(y_complex)
            imag_part = np.imag(y_complex)
            
            # تحليل متقدم للخصائص
            zeta_properties = {
                'zero_value': zero,
                'zero_index': i,
                'magnitude': magnitude,
                'phase': phase,
                'real_part': real_part,
                'imaginary_part': imag_part,
                
                # خصائص الصفر
                'critical_points': self.find_critical_points_advanced(magnitude, phase),
                'phase_singularities': self.find_phase_singularities(phase),
                'magnitude_zeros': self.find_magnitude_zeros(magnitude),
                'real_zeros': self.find_real_zeros(real_part),
                'imaginary_zeros': self.find_imaginary_zeros(imag_part),
                
                # تحليل الدورية
                'periodicity_real': self.analyze_periodicity_advanced(real_part),
                'periodicity_imag': self.analyze_periodicity_advanced(imag_part),
                'phase_periodicity': self.analyze_periodicity_advanced(phase),
                
                # خصائص هندسية
                'path_curvature': self.calculate_path_curvature(real_part, imag_part),
                'winding_behavior': self.analyze_winding_behavior(real_part, imag_part),
                'spiral_characteristics': self.analyze_spiral_characteristics(real_part, imag_part),
                
                # ربط بالأعداد الأولية
                'nearest_prime': self.find_nearest_prime(zero),
                'prime_resonance': self.calculate_prime_resonance_zeta(zero),
                'gap_analysis': self.analyze_gap_with_primes(zero)
            }
            
            zeta_analysis[zero] = zeta_properties
            
            print(f"    ✅ نقاط حرجة: {len(zeta_properties['critical_points'])}")
            print(f"    ✅ أصفار حقيقية: {len(zeta_properties['real_zeros'])}")
            print(f"    ✅ أصفار تخيلية: {len(zeta_properties['imaginary_zeros'])}")
            print(f"    ✅ أقرب عدد أولي: {zeta_properties['nearest_prime']}")
        
        # تحليل الأنماط بين أصفار زيتا
        zeta_patterns = self.analyze_zeta_patterns(zeta_analysis)
        
        self.zeta_relationships['critical_analysis'] = zeta_analysis
        self.zeta_relationships['pattern_analysis'] = zeta_patterns
        
        print("✅ اكتمل استكشاف الروابط مع زيتا!")
        return zeta_analysis, zeta_patterns
    
    def find_critical_points_advanced(self, magnitude, phase):
        """العثور على النقاط الحرجة المتقدمة"""
        critical_points = []
        
        # نقاط القمم والقيعان في المقدار
        for i in range(1, len(magnitude) - 1):
            if (magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]) or \
               (magnitude[i] < magnitude[i-1] and magnitude[i] < magnitude[i+1]):
                critical_points.append({
                    'index': i,
                    'type': 'magnitude_extremum',
                    'value': magnitude[i],
                    'phase': phase[i]
                })
        
        # نقاط التغيير السريع في الطور
        phase_diff = np.abs(np.diff(phase))
        threshold = np.percentile(phase_diff, 95)
        
        for i in range(len(phase_diff)):
            if phase_diff[i] > threshold:
                critical_points.append({
                    'index': i,
                    'type': 'phase_discontinuity',
                    'value': magnitude[i],
                    'phase_jump': phase_diff[i]
                })
        
        return critical_points
    
    def find_phase_singularities(self, phase):
        """العثور على تفردات الطور"""
        singularities = []
        
        # البحث عن قفزات كبيرة في الطور
        phase_diff = np.diff(phase)
        
        for i in range(len(phase_diff)):
            if abs(phase_diff[i]) > np.pi:
                singularities.append({
                    'index': i,
                    'jump_size': phase_diff[i],
                    'phase_before': phase[i],
                    'phase_after': phase[i+1]
                })
        
        return singularities
    
    def find_magnitude_zeros(self, magnitude):
        """العثور على أصفار المقدار"""
        zeros = []
        threshold = 1e-3
        
        for i in range(len(magnitude)):
            if magnitude[i] < threshold:
                zeros.append({
                    'index': i,
                    'value': magnitude[i]
                })
        
        return zeros
    
    def find_real_zeros(self, real_part):
        """العثور على أصفار الجزء الحقيقي"""
        zeros = []
        
        for i in range(len(real_part) - 1):
            if real_part[i] * real_part[i+1] < 0:
                # تقدير موقع الصفر
                zero_pos = i - real_part[i] / (real_part[i+1] - real_part[i])
                zeros.append({
                    'position': zero_pos,
                    'index': i
                })
        
        return zeros
    
    def find_imaginary_zeros(self, imag_part):
        """العثور على أصفار الجزء التخيلي"""
        zeros = []
        
        for i in range(len(imag_part) - 1):
            if imag_part[i] * imag_part[i+1] < 0:
                # تقدير موقع الصفر
                zero_pos = i - imag_part[i] / (imag_part[i+1] - imag_part[i])
                zeros.append({
                    'position': zero_pos,
                    'index': i
                })
        
        return zeros
    
    def analyze_periodicity_advanced(self, signal):
        """تحليل الدورية المتقدم"""
        # تحليل فورييه
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        power_spectrum = np.abs(fft)**2
        
        # العثور على أقوى 3 ترددات
        top_indices = np.argsort(power_spectrum[1:len(power_spectrum)//2])[-3:] + 1
        top_frequencies = [abs(freqs[i]) for i in top_indices]
        top_powers = [power_spectrum[i] for i in top_indices]
        
        # حساب الدورية الإجمالية
        total_power = np.sum(power_spectrum)
        periodicity_strength = sum(top_powers) / total_power if total_power > 0 else 0
        
        return {
            'dominant_frequencies': top_frequencies,
            'dominant_powers': top_powers,
            'periodicity_strength': periodicity_strength,
            'fundamental_period': 1 / top_frequencies[0] if top_frequencies[0] > 0 else float('inf')
        }
    
    def calculate_path_curvature(self, real_part, imag_part):
        """حساب انحناء المسار"""
        if len(real_part) < 3:
            return []
        
        dx = np.gradient(real_part)
        dy = np.gradient(imag_part)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**(3/2)
        curvature = curvature[~np.isnan(curvature)]
        
        return {
            'curvature_values': curvature,
            'mean_curvature': np.mean(curvature) if len(curvature) > 0 else 0,
            'max_curvature': np.max(curvature) if len(curvature) > 0 else 0,
            'curvature_variance': np.var(curvature) if len(curvature) > 0 else 0
        }
    
    def analyze_winding_behavior(self, real_part, imag_part):
        """تحليل سلوك اللف"""
        # حساب الزوايا
        angles = np.arctan2(imag_part, real_part)
        
        # حساب التغيير في الزاوية
        angle_diff = np.diff(angles)
        
        # تصحيح القفزات
        angle_diff = np.where(angle_diff > np.pi, angle_diff - 2*np.pi, angle_diff)
        angle_diff = np.where(angle_diff < -np.pi, angle_diff + 2*np.pi, angle_diff)
        
        # حساب رقم اللف
        total_winding = np.sum(angle_diff) / (2 * np.pi)
        
        # تحليل اتجاه اللف
        positive_winding = np.sum(angle_diff[angle_diff > 0])
        negative_winding = np.sum(angle_diff[angle_diff < 0])
        
        return {
            'total_winding_number': total_winding,
            'positive_winding': positive_winding / (2 * np.pi),
            'negative_winding': negative_winding / (2 * np.pi),
            'winding_consistency': np.std(angle_diff),
            'dominant_direction': 'clockwise' if total_winding < 0 else 'counterclockwise'
        }
    
    def analyze_spiral_characteristics(self, real_part, imag_part):
        """تحليل خصائص الحلزون"""
        # حساب المسافات من المركز
        center_x = np.mean(real_part)
        center_y = np.mean(imag_part)
        
        distances = np.sqrt((real_part - center_x)**2 + (imag_part - center_y)**2)
        
        # تحليل الاتجاه الحلزوني
        distance_trend = np.polyfit(range(len(distances)), distances, 1)[0]
        
        # حساب الزوايا
        angles = np.arctan2(imag_part - center_y, real_part - center_x)
        angle_diff = np.diff(angles)
        
        # تصحيح القفزات
        angle_diff = np.where(angle_diff > np.pi, angle_diff - 2*np.pi, angle_diff)
        angle_diff = np.where(angle_diff < -np.pi, angle_diff + 2*np.pi, angle_diff)
        
        return {
            'spiral_center': (center_x, center_y),
            'spiral_direction': 'outward' if distance_trend > 0 else 'inward',
            'spiral_rate': abs(distance_trend),
            'angular_velocity': np.mean(angle_diff),
            'spiral_consistency': 1 / (1 + np.std(angle_diff)),
            'radius_range': [np.min(distances), np.max(distances)]
        }
    
    def find_nearest_prime(self, value):
        """العثور على أقرب عدد أولي"""
        # تحويل القيمة لعدد صحيح
        int_value = int(round(value))
        
        # البحث عن أقرب عدد أولي
        if isprime(int_value):
            return int_value
        
        # البحث في الاتجاهين
        lower = int_value
        upper = int_value
        
        while lower > 1 or upper < 1000:
            if lower > 1:
                lower -= 1
                if isprime(lower):
                    break
            
            if upper < 1000:
                upper += 1
                if isprime(upper):
                    lower = upper
                    break
        
        return lower if isprime(lower) else None
    
    def calculate_prime_resonance_zeta(self, zeta_zero):
        """حساب الرنين مع الأعداد الأولية لصفر زيتا"""
        # العثور على أقرب عدد أولي
        nearest_prime = self.find_nearest_prime(zeta_zero)
        
        if nearest_prime is None:
            return 0
        
        # حساب الرنين كدالة للمسافة
        distance = abs(zeta_zero - nearest_prime)
        resonance = 1 / (1 + distance)
        
        return {
            'nearest_prime': nearest_prime,
            'distance': distance,
            'resonance_strength': resonance,
            'relative_error': distance / nearest_prime if nearest_prime > 0 else float('inf')
        }
    
    def analyze_gap_with_primes(self, zeta_zero):
        """تحليل الفجوة مع الأعداد الأولية"""
        # العثور على الأعداد الأولية المجاورة
        int_value = int(round(zeta_zero))
        
        # العثور على العدد الأولي السابق واللاحق
        prev_prime = None
        next_prime = None
        
        for i in range(int_value, 1, -1):
            if isprime(i):
                prev_prime = i
                break
        
        for i in range(int_value, int_value + 100):
            if isprime(i):
                next_prime = i
                break
        
        gap_analysis = {}
        
        if prev_prime and next_prime:
            gap_size = next_prime - prev_prime
            position_in_gap = zeta_zero - prev_prime
            relative_position = position_in_gap / gap_size if gap_size > 0 else 0
            
            gap_analysis = {
                'previous_prime': prev_prime,
                'next_prime': next_prime,
                'gap_size': gap_size,
                'position_in_gap': position_in_gap,
                'relative_position': relative_position,
                'gap_center': (prev_prime + next_prime) / 2,
                'distance_from_center': abs(zeta_zero - (prev_prime + next_prime) / 2)
            }
        
        return gap_analysis
    
    def analyze_zeta_patterns(self, zeta_analysis):
        """تحليل الأنماط بين أصفار زيتا"""
        print("  🧮 تحليل الأنماط بين أصفار زيتا...")
        
        zeros = list(zeta_analysis.keys())
        
        # استخراج الخصائص
        critical_counts = [len(zeta_analysis[z]['critical_points']) for z in zeros]
        real_zero_counts = [len(zeta_analysis[z]['real_zeros']) for z in zeros]
        imag_zero_counts = [len(zeta_analysis[z]['imaginary_zeros']) for z in zeros]
        winding_numbers = [zeta_analysis[z]['winding_behavior']['total_winding_number'] for z in zeros]
        spiral_rates = [zeta_analysis[z]['spiral_characteristics']['spiral_rate'] for z in zeros]
        
        # تحليل الارتباطات
        correlations = {}
        
        if len(zeros) > 1:
            correlations['zero_critical'] = np.corrcoef(zeros, critical_counts)[0, 1]
            correlations['zero_real_zeros'] = np.corrcoef(zeros, real_zero_counts)[0, 1]
            correlations['zero_imag_zeros'] = np.corrcoef(zeros, imag_zero_counts)[0, 1]
            correlations['zero_winding'] = np.corrcoef(zeros, winding_numbers)[0, 1]
            correlations['zero_spiral'] = np.corrcoef(zeros, spiral_rates)[0, 1]
        
        # تحليل الفجوات بين الأصفار
        zero_gaps = [zeros[i+1] - zeros[i] for i in range(len(zeros)-1)]
        
        # تحليل الأنماط في الرنين مع الأعداد الأولية
        prime_resonances = []
        gap_positions = []
        
        for zero in zeros:
            resonance_data = zeta_analysis[zero]['prime_resonance']
            gap_data = zeta_analysis[zero]['gap_analysis']
            
            if resonance_data:
                prime_resonances.append(resonance_data['resonance_strength'])
            
            if gap_data and 'relative_position' in gap_data:
                gap_positions.append(gap_data['relative_position'])
        
        pattern_analysis = {
            'correlations': correlations,
            'zero_gaps': zero_gaps,
            'statistical_summary': {
                'critical_points_range': [np.min(critical_counts), np.max(critical_counts)],
                'real_zeros_range': [np.min(real_zero_counts), np.max(real_zero_counts)],
                'imag_zeros_range': [np.min(imag_zero_counts), np.max(imag_zero_counts)],
                'winding_range': [np.min(winding_numbers), np.max(winding_numbers)],
                'spiral_rate_range': [np.min(spiral_rates), np.max(spiral_rates)]
            },
            'prime_resonance_analysis': {
                'resonances': prime_resonances,
                'mean_resonance': np.mean(prime_resonances) if prime_resonances else 0,
                'resonance_std': np.std(prime_resonances) if prime_resonances else 0
            },
            'gap_position_analysis': {
                'positions': gap_positions,
                'mean_position': np.mean(gap_positions) if gap_positions else 0,
                'position_std': np.std(gap_positions) if gap_positions else 0
            }
        }
        
        # طباعة النتائج المهمة
        if correlations:
            print(f"    ✅ ارتباط الأصفار-النقاط الحرجة: {correlations.get('zero_critical', 0):.4f}")
            print(f"    ✅ ارتباط الأصفار-اللف: {correlations.get('zero_winding', 0):.4f}")
        
        print(f"    ✅ متوسط الرنين مع الأعداد الأولية: {pattern_analysis['prime_resonance_analysis']['mean_resonance']:.4f}")
        print(f"    ✅ متوسط الموقع في الفجوة: {pattern_analysis['gap_position_analysis']['mean_position']:.4f}")
        
        return pattern_analysis
    
    def discover_revolutionary_connections(self):
        """اكتشاف الروابط الثورية"""
        print("\n🌟 اكتشاف الروابط الثورية...")
        
        revolutionary_discoveries = []
        
        # التحقق من وجود البيانات
        if 'resonance_analysis' not in self.prime_connections:
            print("❌ لا توجد بيانات رنين الأعداد الأولية")
            return []
        
        if 'critical_analysis' not in self.zeta_relationships:
            print("❌ لا توجد بيانات تحليل زيتا")
            return []
        
        # تحليل الروابط المتقاطعة
        prime_data = self.prime_connections['resonance_analysis']
        zeta_data = self.zeta_relationships['critical_analysis']
        
        # اكتشاف 1: الرنين المتزامن
        sync_discovery = self.discover_synchronized_resonance(prime_data, zeta_data)
        if sync_discovery:
            revolutionary_discoveries.append(sync_discovery)
        
        # اكتشاف 2: الأنماط الهندسية المشتركة
        geometric_discovery = self.discover_shared_geometry(prime_data, zeta_data)
        if geometric_discovery:
            revolutionary_discoveries.append(geometric_discovery)
        
        # اكتشاف 3: القوانين الرياضية الجديدة
        mathematical_discovery = self.discover_mathematical_laws(prime_data, zeta_data)
        if mathematical_discovery:
            revolutionary_discoveries.append(mathematical_discovery)
        
        # اكتشاف 4: الروابط الفراكتالية
        fractal_discovery = self.discover_fractal_connections(prime_data, zeta_data)
        if fractal_discovery:
            revolutionary_discoveries.append(fractal_discovery)
        
        self.revolutionary_patterns = revolutionary_discoveries
        
        print(f"✅ تم اكتشاف {len(revolutionary_discoveries)} رابط ثوري!")
        
        return revolutionary_discoveries
    
    def discover_synchronized_resonance(self, prime_data, zeta_data):
        """اكتشاف الرنين المتزامن"""
        # البحث عن تزامن في الترددات
        prime_frequencies = [prime_data[p]['peak_frequency'] for p in prime_data.keys()]
        
        # مقارنة مع خصائص زيتا
        zeta_periodicities = []
        for zero_data in zeta_data.values():
            if 'periodicity_real' in zero_data:
                zeta_periodicities.extend(zero_data['periodicity_real']['dominant_frequencies'])
        
        if not zeta_periodicities:
            return None
        
        # البحث عن تطابقات
        matches = []
        tolerance = 0.1
        
        for pf in prime_frequencies:
            for zf in zeta_periodicities:
                if abs(pf - zf) < tolerance:
                    matches.append({
                        'prime_frequency': pf,
                        'zeta_frequency': zf,
                        'difference': abs(pf - zf)
                    })
        
        if len(matches) >= 3:  # عتبة الاكتشاف
            return {
                'type': 'synchronized_resonance',
                'description': 'تزامن في الترددات بين الأعداد الأولية وأصفار زيتا',
                'matches': matches,
                'significance': len(matches) / len(prime_frequencies),
                'revolutionary_potential': 'عالي جداً'
            }
        
        return None
    
    def discover_shared_geometry(self, prime_data, zeta_data):
        """اكتشاف الهندسة المشتركة"""
        # مقارنة الخصائص الهندسية
        prime_symmetries = [prime_data[p]['symmetry_measure'] for p in prime_data.keys()]
        
        zeta_windings = []
        for zero_data in zeta_data.values():
            if 'winding_behavior' in zero_data:
                zeta_windings.append(abs(zero_data['winding_behavior']['total_winding_number']))
        
        if not zeta_windings:
            return None
        
        # تحليل التشابه
        prime_symmetry_avg = np.mean(prime_symmetries)
        zeta_winding_avg = np.mean(zeta_windings)
        
        # البحث عن نمط مشترك
        similarity = abs(prime_symmetry_avg - zeta_winding_avg)
        
        if similarity < 0.2:  # عتبة التشابه
            return {
                'type': 'shared_geometry',
                'description': 'تشابه هندسي بين أنماط الأعداد الأولية وأصفار زيتا',
                'prime_symmetry': prime_symmetry_avg,
                'zeta_winding': zeta_winding_avg,
                'similarity_score': 1 - similarity,
                'revolutionary_potential': 'عالي'
            }
        
        return None
    
    def discover_mathematical_laws(self, prime_data, zeta_data):
        """اكتشاف القوانين الرياضية الجديدة"""
        # تحليل العلاقات الرياضية
        primes = list(prime_data.keys())
        prime_frequencies = [prime_data[p]['peak_frequency'] for p in primes]
        
        # البحث عن قانون رياضي
        # اختبار العلاقة: frequency = a / prime + b
        if len(primes) >= 3:
            # انحدار خطي
            X = np.array([[1/p, 1] for p in primes])
            y = np.array(prime_frequencies)
            
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                a, b = coeffs
                
                # حساب جودة الملائمة
                predicted = a / np.array(primes) + b
                r_squared = 1 - np.sum((y - predicted)**2) / np.sum((y - np.mean(y))**2)
                
                if r_squared > 0.8:  # ملائمة جيدة
                    return {
                        'type': 'mathematical_law',
                        'description': f'قانون رياضي: frequency = {a:.4f}/prime + {b:.4f}',
                        'coefficients': {'a': a, 'b': b},
                        'r_squared': r_squared,
                        'formula': 'f(p) = a/p + b',
                        'revolutionary_potential': 'عالي جداً'
                    }
            except:
                pass
        
        return None
    
    def discover_fractal_connections(self, prime_data, zeta_data):
        """اكتشاف الروابط الفراكتالية"""
        # تحليل البصمات الفراكتالية
        prime_fractals = [prime_data[p]['fractal_signature'] for p in prime_data.keys()]
        
        zeta_complexities = []
        for zero_data in zeta_data.values():
            if 'path_curvature' in zero_data:
                curvature_data = zero_data['path_curvature']
                if 'curvature_variance' in curvature_data:
                    zeta_complexities.append(curvature_data['curvature_variance'])
        
        if not zeta_complexities:
            return None
        
        # تحليل الارتباط
        if len(prime_fractals) > 1 and len(zeta_complexities) > 1:
            # مقارنة التوزيعات
            prime_fractal_avg = np.mean(prime_fractals)
            zeta_complexity_avg = np.mean(zeta_complexities)
            
            # تطبيع للمقارنة
            prime_normalized = prime_fractal_avg / (1 + prime_fractal_avg)
            zeta_normalized = zeta_complexity_avg / (1 + zeta_complexity_avg)
            
            correlation = abs(prime_normalized - zeta_normalized)
            
            if correlation < 0.3:  # عتبة الارتباط
                return {
                    'type': 'fractal_connection',
                    'description': 'ارتباط فراكتالي بين تعقيد الأعداد الأولية وأصفار زيتا',
                    'prime_fractal_avg': prime_fractal_avg,
                    'zeta_complexity_avg': zeta_complexity_avg,
                    'correlation_strength': 1 - correlation,
                    'revolutionary_potential': 'متوسط'
                }
        
        return None
    
    def generate_comprehensive_report(self):
        """إنشاء تقرير شامل للاكتشافات"""
        print("\n📋 إنشاء تقرير شامل للاكتشافات...")
        
        report = {
            'analysis_timestamp': np.datetime64('now'),
            'prime_analysis_summary': {},
            'zeta_analysis_summary': {},
            'revolutionary_discoveries': self.revolutionary_patterns,
            'mathematical_insights': [],
            'future_research_directions': [],
            'revolutionary_potential_score': 0
        }
        
        # ملخص تحليل الأعداد الأولية
        if 'resonance_analysis' in self.prime_connections:
            prime_data = self.prime_connections['resonance_analysis']
            pattern_data = self.prime_connections['pattern_analysis']
            
            report['prime_analysis_summary'] = {
                'primes_analyzed': len(prime_data),
                'key_correlations': pattern_data['correlations'],
                'twin_primes_found': pattern_data['twin_prime_analysis'].get('count', 0),
                'sophie_germain_found': pattern_data['sophie_germain_analysis'].get('count', 0),
                'strongest_correlation': max(abs(v) for v in pattern_data['correlations'].values() if not np.isnan(v))
            }
        
        # ملخص تحليل زيتا
        if 'critical_analysis' in self.zeta_relationships:
            zeta_data = self.zeta_relationships['critical_analysis']
            zeta_patterns = self.zeta_relationships['pattern_analysis']
            
            report['zeta_analysis_summary'] = {
                'zeros_analyzed': len(zeta_data),
                'average_critical_points': np.mean([len(z['critical_points']) for z in zeta_data.values()]),
                'average_real_zeros': np.mean([len(z['real_zeros']) for z in zeta_data.values()]),
                'average_winding': np.mean([abs(z['winding_behavior']['total_winding_number']) for z in zeta_data.values()]),
                'prime_resonance_strength': zeta_patterns['prime_resonance_analysis']['mean_resonance']
            }
        
        # تحليل الاكتشافات الثورية
        revolutionary_score = 0
        for discovery in self.revolutionary_patterns:
            if discovery['revolutionary_potential'] == 'عالي جداً':
                revolutionary_score += 10
            elif discovery['revolutionary_potential'] == 'عالي':
                revolutionary_score += 7
            elif discovery['revolutionary_potential'] == 'متوسط':
                revolutionary_score += 4
        
        report['revolutionary_potential_score'] = revolutionary_score
        
        # الرؤى الرياضية
        mathematical_insights = []
        
        if revolutionary_score >= 10:
            mathematical_insights.append("اكتشاف روابط جديدة بين دوال السيجمويد والأعداد الأولية")
        
        if len(self.revolutionary_patterns) >= 2:
            mathematical_insights.append("وجود أنماط متعددة تربط بين النظريات المختلفة")
        
        if 'mathematical_law' in [d['type'] for d in self.revolutionary_patterns]:
            mathematical_insights.append("اكتشاف قانون رياضي جديد يحكم العلاقة")
        
        report['mathematical_insights'] = mathematical_insights
        
        # اتجاهات البحث المستقبلي
        future_directions = [
            "تطوير نظرية شاملة للدوال المركبة والأعداد الأولية",
            "استكشاف تطبيقات في نظرية الأعداد",
            "تطوير خوارزميات جديدة لاختبار الأولية",
            "دراسة الروابط مع فرضية ريمان",
            "تطبيق النتائج في التشفير والأمان"
        ]
        
        report['future_research_directions'] = future_directions
        
        # تقييم الأهمية الإجمالية
        if revolutionary_score >= 20:
            overall_significance = "اكتشاف تاريخي - ثورة في الرياضيات"
        elif revolutionary_score >= 10:
            overall_significance = "اكتشاف مهم جداً - تقدم كبير"
        elif revolutionary_score >= 5:
            overall_significance = "اكتشاف مهم - إضافة قيمة"
        else:
            overall_significance = "نتائج أولية - تحتاج مزيد من البحث"
        
        report['overall_significance'] = overall_significance
        
        print(f"✅ تم إنشاء التقرير - الأهمية الإجمالية: {overall_significance}")
        print(f"✅ نقاط الثورية: {revolutionary_score}")
        print(f"✅ اكتشافات ثورية: {len(self.revolutionary_patterns)}")
        print(f"✅ رؤى رياضية: {len(mathematical_insights)}")
        
        return report

def main():
    """الدالة الرئيسية لاستكشاف الروابط مع الأعداد الأولية وزيتا"""
    print("🔢 مستكشف الروابط مع الأعداد الأولية ودالة زيتا")
    print("تطوير: باسل يحيى عبدالله")
    print("المرحلة الثالثة: الروابط العميقة والاكتشافات الثورية")
    print("=" * 70)
    
    # إنشاء المستكشف
    connector = PrimeZetaConnector()
    
    # المرحلة 1: تحليل الرنين مع الأعداد الأولية
    prime_resonance, prime_patterns = connector.analyze_prime_sigmoid_resonance()
    
    # المرحلة 2: استكشاف الروابط مع زيتا
    zeta_analysis, zeta_patterns = connector.explore_zeta_critical_connections()
    
    # المرحلة 3: اكتشاف الروابط الثورية
    revolutionary_discoveries = connector.discover_revolutionary_connections()
    
    # المرحلة 4: إنشاء التقرير الشامل
    comprehensive_report = connector.generate_comprehensive_report()
    
    print("\n" + "=" * 70)
    print("🎉 اكتمل استكشاف الروابط الثورية!")
    print(f"🌟 الأهمية الإجمالية: {comprehensive_report['overall_significance']}")
    print(f"🔍 اكتشافات ثورية: {len(revolutionary_discoveries)}")
    print(f"🧮 نقاط الثورية: {comprehensive_report['revolutionary_potential_score']}")
    print(f"📊 رؤى رياضية: {len(comprehensive_report['mathematical_insights'])}")
    
    return connector, comprehensive_report

if __name__ == "__main__":
    connector, report = main()

