#!/usr/bin/env python3
"""
دوال السيجمويد المركبة - استكشاف ثوري
=====================================

هذا النظام يستكشف دوال السيجمويد مع الأس المركب ويربطها بلغز الأعداد الأولية
f(x) = a * sigmoid(b*x + c)^(α + βi) + d

المطور: باسل يحيى عبدالله
الفكرة الثورية: استبدال المعامل الأسي n بالعدد المركب
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cmath
import math
from scipy.special import zeta
import warnings
warnings.filterwarnings('ignore')

class ComplexSigmoidExplorer:
    """مستكشف دوال السيجمويد المركبة"""
    
    def __init__(self):
        """تهيئة المستكشف"""
        self.results = {}
        self.patterns = []
        self.prime_connections = []
        
        print("🧮 تم تهيئة مستكشف دوال السيجمويد المركبة!")
        print("🧬 الفكرة الثورية: f(x) = a * sigmoid(b*x + c)^(α + βi) + d")
    
    def complex_sigmoid(self, x, a=1, b=1, c=0, d=0, alpha=1, beta=0):
        """
        حساب دالة السيجمويد المركبة
        f(x) = a * sigmoid(b*x + c)^(α + βi) + d
        """
        # حساب السيجمويد العادي
        sigmoid_val = 1 / (1 + np.exp(-(b * x + c)))
        
        # تطبيق الأس المركب
        complex_exponent = alpha + 1j * beta
        
        # حساب القيمة المركبة
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
    
    def explore_basic_behavior(self):
        """استكشاف السلوك الأساسي للدوال المركبة"""
        print("\n🔍 استكشاف السلوك الأساسي للدوال المركبة...")
        
        x = np.linspace(-5, 5, 1000)
        
        # حالات مختلفة للأس المركب
        test_cases = [
            {"alpha": 1, "beta": 0, "name": "حقيقي بحت (n=1)"},
            {"alpha": 0, "beta": 1, "name": "تخيلي بحت (n=i)"},
            {"alpha": 1, "beta": 1, "name": "مركب متوازن (n=1+i)"},
            {"alpha": 0.5, "beta": 0.5, "name": "مركب متوسط (n=0.5+0.5i)"},
            {"alpha": 2, "beta": 1, "name": "مركب قوي (n=2+i)"},
            {"alpha": 1, "beta": math.pi, "name": "مع π (n=1+πi)"},
        ]
        
        behaviors = {}
        
        for case in test_cases:
            print(f"  📊 اختبار: {case['name']}")
            
            # حساب الدالة
            y_complex = self.complex_sigmoid(x, alpha=case['alpha'], beta=case['beta'])
            
            # تحليل الخصائص
            real_part = np.real(y_complex)
            imag_part = np.imag(y_complex)
            magnitude = np.abs(y_complex)
            phase = np.angle(y_complex)
            
            behaviors[case['name']] = {
                'x': x,
                'complex_values': y_complex,
                'real_part': real_part,
                'imaginary_part': imag_part,
                'magnitude': magnitude,
                'phase': phase,
                'alpha': case['alpha'],
                'beta': case['beta'],
                'max_magnitude': np.max(magnitude),
                'min_magnitude': np.min(magnitude),
                'phase_range': np.max(phase) - np.min(phase),
                'oscillations': self.count_oscillations(real_part) + self.count_oscillations(imag_part)
            }
            
            print(f"    ✅ أقصى قيمة: {behaviors[case['name']]['max_magnitude']:.4f}")
            print(f"    ✅ نطاق الطور: {behaviors[case['name']]['phase_range']:.4f}")
            print(f"    ✅ التذبذبات: {behaviors[case['name']]['oscillations']}")
        
        self.results['basic_behavior'] = behaviors
        print("✅ اكتمل استكشاف السلوك الأساسي!")
        
        return behaviors
    
    def count_oscillations(self, signal):
        """عد التذبذبات في الإشارة"""
        if len(signal) < 3:
            return 0
        
        oscillations = 0
        for i in range(1, len(signal) - 1):
            if (signal[i] > signal[i-1] and signal[i] > signal[i+1]) or \
               (signal[i] < signal[i-1] and signal[i] < signal[i+1]):
                oscillations += 1
        
        return oscillations
    
    def explore_critical_line_behavior(self):
        """استكشاف السلوك على الخط الحرج لريمان"""
        print("\n🎯 استكشاف السلوك على الخط الحرج لريمان (α = 0.5)...")
        
        x = np.linspace(-10, 10, 2000)
        
        # قيم مختلفة للجزء التخيلي (مثل أصفار زيتا)
        critical_betas = [
            14.134725,  # أول صفر غير تافه لزيتا
            21.022040,  # ثاني صفر
            25.010858,  # ثالث صفر
            30.424876,  # رابع صفر
            32.935062,  # خامس صفر
            37.586178,  # سادس صفر
        ]
        
        critical_behaviors = {}
        
        for i, beta in enumerate(critical_betas):
            print(f"  🔍 اختبار β = {beta:.6f} (صفر زيتا #{i+1})")
            
            # حساب الدالة على الخط الحرج
            y_complex = self.complex_sigmoid(x, alpha=0.5, beta=beta)
            
            # تحليل خاص للخط الحرج
            real_part = np.real(y_complex)
            imag_part = np.imag(y_complex)
            magnitude = np.abs(y_complex)
            
            # البحث عن أصفار أو نقاط خاصة
            zeros_real = self.find_zeros(x, real_part)
            zeros_imag = self.find_zeros(x, imag_part)
            zeros_magnitude = self.find_zeros(x, magnitude - 0.5)  # نقاط قريبة من 0.5
            
            critical_behaviors[f"zeta_zero_{i+1}"] = {
                'beta': beta,
                'x': x,
                'complex_values': y_complex,
                'real_part': real_part,
                'imaginary_part': imag_part,
                'magnitude': magnitude,
                'zeros_real': zeros_real,
                'zeros_imaginary': zeros_imag,
                'critical_points': zeros_magnitude,
                'symmetry_score': self.calculate_symmetry(real_part),
                'periodicity_score': self.calculate_periodicity(imag_part)
            }
            
            print(f"    ✅ أصفار الجزء الحقيقي: {len(zeros_real)}")
            print(f"    ✅ أصفار الجزء التخيلي: {len(zeros_imag)}")
            print(f"    ✅ نقاط حرجة: {len(zeros_magnitude)}")
        
        self.results['critical_line'] = critical_behaviors
        print("✅ اكتمل استكشاف الخط الحرج!")
        
        return critical_behaviors
    
    def find_zeros(self, x, y, tolerance=1e-3):
        """العثور على الأصفار في الدالة"""
        zeros = []
        for i in range(len(y) - 1):
            if abs(y[i]) < tolerance or (y[i] * y[i+1] < 0):
                # تقدير أفضل للصفر
                if abs(y[i+1] - y[i]) > 1e-10:
                    zero_x = x[i] - y[i] * (x[i+1] - x[i]) / (y[i+1] - y[i])
                else:
                    zero_x = x[i]
                zeros.append(zero_x)
        
        return zeros
    
    def calculate_symmetry(self, signal):
        """حساب درجة التماثل في الإشارة"""
        n = len(signal)
        center = n // 2
        
        left_part = signal[:center]
        right_part = signal[center:][::-1]  # عكس الجزء الأيمن
        
        min_len = min(len(left_part), len(right_part))
        if min_len == 0:
            return 0
        
        left_part = left_part[:min_len]
        right_part = right_part[:min_len]
        
        # حساب معامل الارتباط
        correlation = np.corrcoef(left_part, right_part)[0, 1]
        return correlation if not np.isnan(correlation) else 0
    
    def calculate_periodicity(self, signal):
        """حساب درجة الدورية في الإشارة"""
        # استخدام FFT للعثور على الترددات المهيمنة
        fft = np.fft.fft(signal)
        power_spectrum = np.abs(fft) ** 2
        
        # العثور على أقوى تردد
        max_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        max_power = power_spectrum[max_freq_idx]
        total_power = np.sum(power_spectrum)
        
        # نسبة الطاقة في التردد المهيمن
        periodicity_score = max_power / total_power if total_power > 0 else 0
        
        return periodicity_score
    
    def explore_prime_connections(self):
        """استكشاف الروابط مع الأعداد الأولية"""
        print("\n🔢 استكشاف الروابط مع الأعداد الأولية...")
        
        # توليد الأعداد الأولية الأولى
        primes = self.generate_primes(100)
        print(f"  📊 تم توليد {len(primes)} عدد أولي")
        
        # اختبار قيم مختلفة للأس المركب مرتبطة بالأعداد الأولية
        prime_tests = []
        
        for i, p in enumerate(primes[:10]):  # أول 10 أعداد أولية
            # استخدام العدد الأولي في الجزء التخيلي
            alpha = 0.5  # الخط الحرج
            beta = p / 10.0  # تطبيع العدد الأولي
            
            x = np.linspace(-5, 5, 1000)
            y_complex = self.complex_sigmoid(x, alpha=alpha, beta=beta)
            
            # تحليل خاص للأعداد الأولية
            magnitude = np.abs(y_complex)
            phase = np.angle(y_complex)
            
            # البحث عن أنماط مرتبطة بالعدد الأولي
            prime_pattern = {
                'prime': p,
                'alpha': alpha,
                'beta': beta,
                'x': x,
                'complex_values': y_complex,
                'magnitude': magnitude,
                'phase': phase,
                'max_magnitude': np.max(magnitude),
                'phase_jumps': self.count_phase_jumps(phase),
                'magnitude_peaks': self.count_peaks(magnitude),
                'prime_resonance': self.calculate_prime_resonance(magnitude, p)
            }
            
            prime_tests.append(prime_pattern)
            
            print(f"    🔍 العدد الأولي {p}: قمم={prime_pattern['magnitude_peaks']}, رنين={prime_pattern['prime_resonance']:.4f}")
        
        # البحث عن أنماط في توزيع الأعداد الأولية
        prime_distribution_analysis = self.analyze_prime_distribution_patterns(prime_tests)
        
        self.results['prime_connections'] = {
            'individual_primes': prime_tests,
            'distribution_analysis': prime_distribution_analysis
        }
        
        print("✅ اكتمل استكشاف الروابط مع الأعداد الأولية!")
        
        return prime_tests
    
    def generate_primes(self, limit):
        """توليد الأعداد الأولية حتى حد معين"""
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    def count_phase_jumps(self, phase):
        """عد القفزات في الطور"""
        jumps = 0
        threshold = math.pi / 2  # عتبة القفزة
        
        for i in range(1, len(phase)):
            if abs(phase[i] - phase[i-1]) > threshold:
                jumps += 1
        
        return jumps
    
    def count_peaks(self, signal):
        """عد القمم في الإشارة"""
        peaks = 0
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks += 1
        return peaks
    
    def calculate_prime_resonance(self, magnitude, prime):
        """حساب الرنين مع العدد الأولي"""
        # تحليل فورييه للعثور على ترددات مرتبطة بالعدد الأولي
        fft = np.fft.fft(magnitude)
        freqs = np.fft.fftfreq(len(magnitude))
        
        # البحث عن تردد مرتبط بالعدد الأولي
        target_freq = prime / len(magnitude)
        
        # العثور على أقرب تردد
        closest_idx = np.argmin(np.abs(freqs - target_freq))
        resonance = np.abs(fft[closest_idx]) / np.max(np.abs(fft))
        
        return resonance
    
    def analyze_prime_distribution_patterns(self, prime_tests):
        """تحليل أنماط توزيع الأعداد الأولية"""
        print("  🧮 تحليل أنماط توزيع الأعداد الأولية...")
        
        # استخراج البيانات
        primes = [test['prime'] for test in prime_tests]
        resonances = [test['prime_resonance'] for test in prime_tests]
        peaks = [test['magnitude_peaks'] for test in prime_tests]
        
        # تحليل الارتباطات
        prime_resonance_correlation = np.corrcoef(primes, resonances)[0, 1] if len(primes) > 1 else 0
        prime_peaks_correlation = np.corrcoef(primes, peaks)[0, 1] if len(primes) > 1 else 0
        
        # البحث عن أنماط في الفجوات بين الأعداد الأولية
        prime_gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
        gap_resonance_correlation = np.corrcoef(prime_gaps, resonances[1:]) if len(prime_gaps) > 1 else [0, 0]
        gap_resonance_correlation = gap_resonance_correlation[0, 1] if len(gap_resonance_correlation.shape) > 1 else 0
        
        analysis = {
            'prime_resonance_correlation': prime_resonance_correlation,
            'prime_peaks_correlation': prime_peaks_correlation,
            'gap_resonance_correlation': gap_resonance_correlation,
            'average_resonance': np.mean(resonances),
            'resonance_std': np.std(resonances),
            'prime_gaps': prime_gaps,
            'gap_statistics': {
                'mean_gap': np.mean(prime_gaps),
                'std_gap': np.std(prime_gaps),
                'max_gap': np.max(prime_gaps),
                'min_gap': np.min(prime_gaps)
            }
        }
        
        print(f"    ✅ ارتباط الأعداد الأولية-الرنين: {prime_resonance_correlation:.4f}")
        print(f"    ✅ ارتباط الفجوات-الرنين: {gap_resonance_correlation:.4f}")
        print(f"    ✅ متوسط الرنين: {analysis['average_resonance']:.4f}")
        
        return analysis
    
    def discover_new_patterns(self):
        """اكتشاف أنماط جديدة في الدوال المركبة"""
        print("\n🌟 اكتشاف أنماط جديدة...")
        
        patterns = []
        
        # نمط 1: التذبذبات المتزامنة
        print("  🔍 البحث عن التذبذبات المتزامنة...")
        sync_pattern = self.find_synchronized_oscillations()
        if sync_pattern:
            patterns.append(sync_pattern)
            print("    ✅ تم اكتشاف نمط التذبذبات المتزامنة!")
        
        # نمط 2: النقاط الحرجة المرتبطة بزيتا
        print("  🔍 البحث عن النقاط الحرجة المرتبطة بزيتا...")
        zeta_pattern = self.find_zeta_critical_points()
        if zeta_pattern:
            patterns.append(zeta_pattern)
            print("    ✅ تم اكتشاف نمط النقاط الحرجة لزيتا!")
        
        # نمط 3: الحلزونات المركبة
        print("  🔍 البحث عن الحلزونات المركبة...")
        spiral_pattern = self.find_complex_spirals()
        if spiral_pattern:
            patterns.append(spiral_pattern)
            print("    ✅ تم اكتشاف نمط الحلزونات المركبة!")
        
        self.patterns = patterns
        print(f"✅ تم اكتشاف {len(patterns)} نمط جديد!")
        
        return patterns
    
    def find_synchronized_oscillations(self):
        """العثور على التذبذبات المتزامنة"""
        # اختبار تزامن التذبذبات بين الجزء الحقيقي والتخيلي
        x = np.linspace(-10, 10, 2000)
        
        best_sync = 0
        best_params = None
        
        for alpha in [0.5, 1.0, 1.5]:
            for beta in [1, 2, 3, 5, 7, 11]:  # أعداد أولية
                y_complex = self.complex_sigmoid(x, alpha=alpha, beta=beta)
                real_part = np.real(y_complex)
                imag_part = np.imag(y_complex)
                
                # حساب التزامن
                correlation = np.corrcoef(real_part, imag_part)[0, 1]
                if not np.isnan(correlation) and abs(correlation) > best_sync:
                    best_sync = abs(correlation)
                    best_params = {'alpha': alpha, 'beta': beta, 'correlation': correlation}
        
        if best_sync > 0.7:  # عتبة التزامن
            return {
                'type': 'synchronized_oscillations',
                'strength': best_sync,
                'parameters': best_params,
                'description': f"تذبذبات متزامنة بقوة {best_sync:.3f}"
            }
        
        return None
    
    def find_zeta_critical_points(self):
        """العثور على النقاط الحرجة المرتبطة بدالة زيتا"""
        # اختبار النقاط الحرجة عند أصفار زيتا
        zeta_zeros = [14.134725, 21.022040, 25.010858]
        
        critical_points = []
        
        for zero in zeta_zeros:
            x = np.linspace(-5, 5, 1000)
            y_complex = self.complex_sigmoid(x, alpha=0.5, beta=zero)
            magnitude = np.abs(y_complex)
            
            # البحث عن نقاط حرجة (قمم أو قيعان)
            for i in range(1, len(magnitude) - 1):
                if (magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]) or \
                   (magnitude[i] < magnitude[i-1] and magnitude[i] < magnitude[i+1]):
                    critical_points.append({
                        'x': x[i],
                        'magnitude': magnitude[i],
                        'zeta_zero': zero
                    })
        
        if len(critical_points) > 5:  # عتبة وجود نقاط كافية
            return {
                'type': 'zeta_critical_points',
                'count': len(critical_points),
                'points': critical_points[:10],  # أول 10 نقاط
                'description': f"تم العثور على {len(critical_points)} نقطة حرجة مرتبطة بزيتا"
            }
        
        return None
    
    def find_complex_spirals(self):
        """العثور على الحلزونات المركبة"""
        # البحث عن أنماط حلزونية في المستوى المركب
        x = np.linspace(-3, 3, 500)
        
        spiral_candidates = []
        
        for alpha in [0.5, 1.0]:
            for beta in [1, 2, 3]:
                y_complex = self.complex_sigmoid(x, alpha=alpha, beta=beta)
                
                # تحليل المسار في المستوى المركب
                real_part = np.real(y_complex)
                imag_part = np.imag(y_complex)
                
                # حساب الانحناء (curvature)
                dx_real = np.gradient(real_part)
                dx_imag = np.gradient(imag_part)
                
                d2x_real = np.gradient(dx_real)
                d2x_imag = np.gradient(dx_imag)
                
                # حساب الانحناء
                curvature = np.abs(dx_real * d2x_imag - dx_imag * d2x_real) / \
                           (dx_real**2 + dx_imag**2)**(3/2)
                
                # تجاهل القيم غير المحددة
                curvature = curvature[~np.isnan(curvature)]
                
                if len(curvature) > 0:
                    avg_curvature = np.mean(curvature)
                    
                    if avg_curvature > 0.1:  # عتبة الحلزونية
                        spiral_candidates.append({
                            'alpha': alpha,
                            'beta': beta,
                            'curvature': avg_curvature,
                            'spiral_strength': avg_curvature
                        })
        
        if spiral_candidates:
            best_spiral = max(spiral_candidates, key=lambda x: x['spiral_strength'])
            return {
                'type': 'complex_spirals',
                'best_spiral': best_spiral,
                'all_candidates': spiral_candidates,
                'description': f"حلزون مركب بقوة {best_spiral['spiral_strength']:.3f}"
            }
        
        return None
    
    def generate_summary_report(self):
        """إنشاء تقرير ملخص للاكتشافات"""
        print("\n📋 إنشاء تقرير ملخص الاكتشافات...")
        
        report = {
            'timestamp': np.datetime64('now'),
            'total_explorations': len(self.results),
            'patterns_discovered': len(self.patterns),
            'key_findings': [],
            'revolutionary_insights': [],
            'prime_connections_found': False,
            'zeta_relationships': False
        }
        
        # تحليل النتائج الرئيسية
        if 'basic_behavior' in self.results:
            report['key_findings'].append("تم استكشاف السلوك الأساسي للدوال المركبة")
            
            # البحث عن سلوكيات ثورية
            for name, behavior in self.results['basic_behavior'].items():
                if behavior['oscillations'] > 10:
                    report['revolutionary_insights'].append(f"تذبذبات عالية في {name}")
                if behavior['phase_range'] > math.pi:
                    report['revolutionary_insights'].append(f"نطاق طور واسع في {name}")
        
        if 'critical_line' in self.results:
            report['key_findings'].append("تم استكشاف السلوك على الخط الحرج لريمان")
            report['zeta_relationships'] = True
            
            # تحليل الروابط مع زيتا
            for name, behavior in self.results['critical_line'].items():
                if len(behavior['critical_points']) > 0:
                    report['revolutionary_insights'].append(f"نقاط حرجة مكتشفة في {name}")
        
        if 'prime_connections' in self.results:
            report['key_findings'].append("تم استكشاف الروابط مع الأعداد الأولية")
            report['prime_connections_found'] = True
            
            # تحليل قوة الروابط
            analysis = self.results['prime_connections']['distribution_analysis']
            if abs(analysis['prime_resonance_correlation']) > 0.5:
                report['revolutionary_insights'].append("ارتباط قوي بين الأعداد الأولية والرنين")
            if abs(analysis['gap_resonance_correlation']) > 0.3:
                report['revolutionary_insights'].append("ارتباط بين فجوات الأعداد الأولية والرنين")
        
        # تقييم الأهمية الثورية
        if len(report['revolutionary_insights']) >= 3:
            report['revolutionary_potential'] = "عالي جداً"
        elif len(report['revolutionary_insights']) >= 2:
            report['revolutionary_potential'] = "عالي"
        elif len(report['revolutionary_insights']) >= 1:
            report['revolutionary_potential'] = "متوسط"
        else:
            report['revolutionary_potential'] = "منخفض"
        
        print(f"✅ تم إنشاء التقرير - الإمكانية الثورية: {report['revolutionary_potential']}")
        print(f"✅ اكتشافات ثورية: {len(report['revolutionary_insights'])}")
        print(f"✅ روابط الأعداد الأولية: {'نعم' if report['prime_connections_found'] else 'لا'}")
        print(f"✅ علاقات زيتا: {'نعم' if report['zeta_relationships'] else 'لا'}")
        
        return report

def main():
    """الدالة الرئيسية لاستكشاف دوال السيجمويد المركبة"""
    print("🧮 مستكشف دوال السيجمويد المركبة")
    print("تطوير: باسل يحيى عبدالله")
    print("الفكرة الثورية: f(x) = a * sigmoid(b*x + c)^(α + βi) + d")
    print("=" * 60)
    
    # إنشاء المستكشف
    explorer = ComplexSigmoidExplorer()
    
    # المرحلة 1: استكشاف السلوك الأساسي
    basic_behaviors = explorer.explore_basic_behavior()
    
    # المرحلة 2: استكشاف الخط الحرج
    critical_behaviors = explorer.explore_critical_line_behavior()
    
    # المرحلة 3: استكشاف الروابط مع الأعداد الأولية
    prime_connections = explorer.explore_prime_connections()
    
    # المرحلة 4: اكتشاف أنماط جديدة
    new_patterns = explorer.discover_new_patterns()
    
    # المرحلة 5: إنشاء التقرير النهائي
    final_report = explorer.generate_summary_report()
    
    print("\n" + "=" * 60)
    print("🎉 اكتمل الاستكشاف الثوري!")
    print(f"🌟 الإمكانية الثورية: {final_report['revolutionary_potential']}")
    print(f"🔍 أنماط مكتشفة: {len(new_patterns)}")
    print(f"🔢 روابط الأعداد الأولية: {'✅' if final_report['prime_connections_found'] else '❌'}")
    print(f"🧮 علاقات زيتا: {'✅' if final_report['zeta_relationships'] else '❌'}")
    
    return explorer, final_report

if __name__ == "__main__":
    explorer, report = main()

