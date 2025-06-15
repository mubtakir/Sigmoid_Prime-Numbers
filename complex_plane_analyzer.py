#!/usr/bin/env python3
"""
تحليل السلوك في المستوى المركب - المرحلة الثانية
===============================================

تحليل متقدم لسلوك دوال السيجمويد المركبة في المستوى المركب
مع تصور بصري ثلاثي الأبعاد واكتشاف الأنماط الهندسية

المطور: باسل يحيى عبدالله
الفكرة الثورية: f(x) = a * sigmoid(b*x + c)^(α + βi) + d
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cmath
import math
from scipy.special import zeta
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# تعيين الخط للعربية
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False

class ComplexPlaneAnalyzer:
    """محلل السلوك في المستوى المركب"""
    
    def __init__(self):
        """تهيئة المحلل"""
        self.analysis_results = {}
        self.visualizations = []
        self.geometric_patterns = []
        
        print("🧮 تم تهيئة محلل المستوى المركب!")
        print("🎯 الهدف: تحليل السلوك الهندسي للدوال المركبة")
    
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
    
    def analyze_complex_trajectories(self):
        """تحليل المسارات في المستوى المركب"""
        print("\n🎯 تحليل المسارات في المستوى المركب...")
        
        x = np.linspace(-5, 5, 1000)
        
        # حالات مختلفة للتحليل
        test_cases = [
            {"alpha": 0, "beta": 1, "name": "تخيلي_بحت", "color": "blue"},
            {"alpha": 1, "beta": 1, "name": "مركب_متوازن", "color": "red"},
            {"alpha": 0.5, "beta": math.pi, "name": "خط_حرج_مع_π", "color": "green"},
            {"alpha": 1, "beta": 14.134725, "name": "صفر_زيتا_أول", "color": "purple"},
            {"alpha": 0.5, "beta": 21.022040, "name": "صفر_زيتا_ثاني", "color": "orange"},
        ]
        
        trajectories = {}
        
        for case in test_cases:
            print(f"  📊 تحليل مسار: {case['name']}")
            
            # حساب الدالة المركبة
            y_complex = self.complex_sigmoid(x, alpha=case['alpha'], beta=case['beta'])
            
            # استخراج المكونات
            real_part = np.real(y_complex)
            imag_part = np.imag(y_complex)
            magnitude = np.abs(y_complex)
            phase = np.angle(y_complex)
            
            # تحليل هندسي متقدم
            trajectory_analysis = {
                'x': x,
                'complex_values': y_complex,
                'real_part': real_part,
                'imaginary_part': imag_part,
                'magnitude': magnitude,
                'phase': phase,
                'parameters': case,
                
                # خصائص هندسية
                'path_length': self.calculate_path_length(real_part, imag_part),
                'curvature': self.calculate_curvature(real_part, imag_part),
                'torsion': self.calculate_torsion(real_part, imag_part, magnitude),
                'winding_number': self.calculate_winding_number(real_part, imag_part),
                'fractal_dimension': self.estimate_fractal_dimension(real_part, imag_part),
                
                # نقاط خاصة
                'critical_points': self.find_critical_points(magnitude, phase),
                'inflection_points': self.find_inflection_points(real_part, imag_part),
                'spiral_centers': self.find_spiral_centers(real_part, imag_part),
                
                # أنماط دورية
                'periodicity': self.analyze_periodicity(phase),
                'symmetries': self.analyze_symmetries(real_part, imag_part),
            }
            
            trajectories[case['name']] = trajectory_analysis
            
            print(f"    ✅ طول المسار: {trajectory_analysis['path_length']:.4f}")
            print(f"    ✅ رقم اللف: {trajectory_analysis['winding_number']:.4f}")
            print(f"    ✅ البعد الفراكتالي: {trajectory_analysis['fractal_dimension']:.4f}")
            print(f"    ✅ نقاط حرجة: {len(trajectory_analysis['critical_points'])}")
        
        self.analysis_results['trajectories'] = trajectories
        print("✅ اكتمل تحليل المسارات!")
        
        return trajectories
    
    def calculate_path_length(self, real_part, imag_part):
        """حساب طول المسار في المستوى المركب"""
        dx = np.diff(real_part)
        dy = np.diff(imag_part)
        path_length = np.sum(np.sqrt(dx**2 + dy**2))
        return path_length
    
    def calculate_curvature(self, real_part, imag_part):
        """حساب الانحناء"""
        if len(real_part) < 3:
            return np.array([])
        
        dx = np.gradient(real_part)
        dy = np.gradient(imag_part)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**(3/2)
        curvature = curvature[~np.isnan(curvature)]
        
        return curvature
    
    def calculate_torsion(self, real_part, imag_part, magnitude):
        """حساب الالتواء في الفضاء ثلاثي الأبعاد"""
        if len(real_part) < 4:
            return np.array([])
        
        # استخدام المقدار كبعد ثالث
        x, y, z = real_part, imag_part, magnitude
        
        dx = np.gradient(x)
        dy = np.gradient(y)
        dz = np.gradient(z)
        
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        d2z = np.gradient(dz)
        
        d3x = np.gradient(d2x)
        d3y = np.gradient(d2y)
        d3z = np.gradient(d2z)
        
        # حساب الالتواء
        cross1 = np.cross(np.column_stack([dx, dy, dz]), 
                         np.column_stack([d2x, d2y, d2z]), axis=1)
        cross2 = np.cross(np.column_stack([d2x, d2y, d2z]), 
                         np.column_stack([d3x, d3y, d3z]), axis=1)
        
        numerator = np.sum(cross1 * cross2, axis=1)
        denominator = np.sum(cross1**2, axis=1)
        
        torsion = numerator / (denominator + 1e-10)
        torsion = torsion[~np.isnan(torsion)]
        
        return torsion
    
    def calculate_winding_number(self, real_part, imag_part):
        """حساب رقم اللف حول الأصل"""
        # تحويل إلى إحداثيات قطبية
        angles = np.arctan2(imag_part, real_part)
        
        # حساب التغيير الإجمالي في الزاوية
        angle_diff = np.diff(angles)
        
        # تصحيح القفزات
        angle_diff = np.where(angle_diff > np.pi, angle_diff - 2*np.pi, angle_diff)
        angle_diff = np.where(angle_diff < -np.pi, angle_diff + 2*np.pi, angle_diff)
        
        total_angle_change = np.sum(angle_diff)
        winding_number = total_angle_change / (2 * np.pi)
        
        return winding_number
    
    def estimate_fractal_dimension(self, real_part, imag_part):
        """تقدير البعد الفراكتالي للمسار"""
        # استخدام طريقة box-counting مبسطة
        points = np.column_stack([real_part, imag_part])
        
        # نطاقات مختلفة للصناديق
        box_sizes = np.logspace(-3, 0, 10)
        counts = []
        
        for box_size in box_sizes:
            # تقسيم المساحة إلى صناديق
            x_min, x_max = np.min(real_part), np.max(real_part)
            y_min, y_max = np.min(imag_part), np.max(imag_part)
            
            x_bins = np.arange(x_min, x_max + box_size, box_size)
            y_bins = np.arange(y_min, y_max + box_size, box_size)
            
            # عد الصناديق التي تحتوي على نقاط
            hist, _, _ = np.histogram2d(real_part, imag_part, bins=[x_bins, y_bins])
            count = np.sum(hist > 0)
            counts.append(count)
        
        # حساب البعد الفراكتالي
        if len(counts) > 1 and np.max(counts) > np.min(counts):
            log_counts = np.log(counts)
            log_sizes = np.log(1/box_sizes)
            
            # الانحدار الخطي
            coeffs = np.polyfit(log_sizes, log_counts, 1)
            fractal_dimension = coeffs[0]
        else:
            fractal_dimension = 1.0
        
        return max(1.0, min(2.0, fractal_dimension))  # تحديد النطاق
    
    def find_critical_points(self, magnitude, phase):
        """العثور على النقاط الحرجة"""
        critical_points = []
        
        # نقاط القمم والقيعان في المقدار
        for i in range(1, len(magnitude) - 1):
            if (magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]) or \
               (magnitude[i] < magnitude[i-1] and magnitude[i] < magnitude[i+1]):
                critical_points.append({
                    'index': i,
                    'magnitude': magnitude[i],
                    'phase': phase[i],
                    'type': 'magnitude_extremum'
                })
        
        # نقاط التغيير السريع في الطور
        phase_diff = np.abs(np.diff(phase))
        threshold = np.percentile(phase_diff, 95)  # أعلى 5%
        
        for i in range(len(phase_diff)):
            if phase_diff[i] > threshold:
                critical_points.append({
                    'index': i,
                    'magnitude': magnitude[i],
                    'phase': phase[i],
                    'type': 'phase_jump'
                })
        
        return critical_points
    
    def find_inflection_points(self, real_part, imag_part):
        """العثور على نقاط الانعطاف"""
        curvature = self.calculate_curvature(real_part, imag_part)
        
        if len(curvature) < 3:
            return []
        
        inflection_points = []
        
        # البحث عن تغييرات في إشارة الانحناء
        curvature_diff = np.diff(curvature)
        
        for i in range(1, len(curvature_diff)):
            if curvature_diff[i-1] * curvature_diff[i] < 0:  # تغيير إشارة
                inflection_points.append({
                    'index': i,
                    'curvature': curvature[i] if i < len(curvature) else 0,
                    'real': real_part[i] if i < len(real_part) else 0,
                    'imag': imag_part[i] if i < len(imag_part) else 0
                })
        
        return inflection_points
    
    def find_spiral_centers(self, real_part, imag_part):
        """العثور على مراكز الحلزونات"""
        spiral_centers = []
        
        # تحليل الحركة الدائرية
        center_x = np.mean(real_part)
        center_y = np.mean(imag_part)
        
        # حساب المسافات من المركز
        distances = np.sqrt((real_part - center_x)**2 + (imag_part - center_y)**2)
        
        # البحث عن أنماط حلزونية (تغيير منتظم في المسافة)
        distance_trend = np.polyfit(range(len(distances)), distances, 1)[0]
        
        if abs(distance_trend) > 0.001:  # عتبة الحلزونية
            # حساب الزوايا
            angles = np.arctan2(imag_part - center_y, real_part - center_x)
            angle_diff = np.diff(angles)
            
            # تصحيح القفزات
            angle_diff = np.where(angle_diff > np.pi, angle_diff - 2*np.pi, angle_diff)
            angle_diff = np.where(angle_diff < -np.pi, angle_diff + 2*np.pi, angle_diff)
            
            # تحليل انتظام الدوران
            angle_consistency = np.std(angle_diff)
            
            if angle_consistency < 0.5:  # عتبة الانتظام
                spiral_centers.append({
                    'center_x': center_x,
                    'center_y': center_y,
                    'spiral_rate': distance_trend,
                    'rotation_consistency': angle_consistency,
                    'direction': 'outward' if distance_trend > 0 else 'inward'
                })
        
        return spiral_centers
    
    def analyze_periodicity(self, phase):
        """تحليل الدورية في الطور"""
        # تحليل فورييه للطور
        fft = np.fft.fft(phase)
        freqs = np.fft.fftfreq(len(phase))
        power_spectrum = np.abs(fft)**2
        
        # العثور على الترددات المهيمنة
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_freq = freqs[dominant_freq_idx]
        dominant_power = power_spectrum[dominant_freq_idx]
        total_power = np.sum(power_spectrum)
        
        periodicity_strength = dominant_power / total_power if total_power > 0 else 0
        period = 1 / abs(dominant_freq) if abs(dominant_freq) > 1e-10 else float('inf')
        
        return {
            'dominant_frequency': dominant_freq,
            'period': period,
            'strength': periodicity_strength,
            'power_spectrum': power_spectrum[:len(power_spectrum)//2]
        }
    
    def analyze_symmetries(self, real_part, imag_part):
        """تحليل التماثلات"""
        symmetries = {}
        
        # تماثل حول المحور الحقيقي
        real_symmetry = np.corrcoef(imag_part, -imag_part[::-1])[0, 1]
        if not np.isnan(real_symmetry):
            symmetries['real_axis'] = real_symmetry
        
        # تماثل حول المحور التخيلي
        imag_symmetry = np.corrcoef(real_part, -real_part[::-1])[0, 1]
        if not np.isnan(imag_symmetry):
            symmetries['imaginary_axis'] = imag_symmetry
        
        # تماثل حول الأصل
        origin_symmetry = np.corrcoef(
            real_part + 1j * imag_part,
            -(real_part[::-1] + 1j * imag_part[::-1])
        )[0, 1]
        if not np.isnan(origin_symmetry):
            symmetries['origin'] = abs(origin_symmetry)
        
        # تماثل دوراني
        angles = np.linspace(0, 2*np.pi, 8)  # اختبار 8 زوايا
        rotational_symmetries = []
        
        for angle in angles[1:]:  # تجاهل الزاوية 0
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotated_real = cos_a * real_part - sin_a * imag_part
            rotated_imag = sin_a * real_part + cos_a * imag_part
            
            correlation = np.corrcoef(
                real_part + 1j * imag_part,
                rotated_real + 1j * rotated_imag
            )[0, 1]
            
            if not np.isnan(correlation):
                rotational_symmetries.append({
                    'angle': angle,
                    'correlation': abs(correlation)
                })
        
        if rotational_symmetries:
            best_rotation = max(rotational_symmetries, key=lambda x: x['correlation'])
            symmetries['rotational'] = best_rotation
        
        return symmetries
    
    def create_complex_plane_visualization(self):
        """إنشاء تصور للمستوى المركب"""
        print("\n🎨 إنشاء تصور المستوى المركب...")
        
        if 'trajectories' not in self.analysis_results:
            print("❌ لا توجد بيانات مسارات للتصور")
            return None
        
        # إنشاء شبكة من الرسوم البيانية
        fig = plt.figure(figsize=(20, 15))
        
        # الرسم الرئيسي: جميع المسارات في المستوى المركب
        ax1 = plt.subplot(2, 3, 1)
        
        trajectories = self.analysis_results['trajectories']
        
        for name, traj in trajectories.items():
            color = traj['parameters']['color']
            real_part = traj['real_part']
            imag_part = traj['imaginary_part']
            
            ax1.plot(real_part, imag_part, color=color, linewidth=2, 
                    label=name.replace('_', ' '), alpha=0.8)
            
            # إضافة نقاط البداية والنهاية
            ax1.scatter(real_part[0], imag_part[0], color=color, s=100, marker='o')
            ax1.scatter(real_part[-1], imag_part[-1], color=color, s=100, marker='s')
        
        ax1.set_xlabel('Real Part')
        ax1.set_ylabel('Imaginary Part')
        ax1.set_title('Complex Plane Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # الرسم الثاني: المقدار مقابل الطور
        ax2 = plt.subplot(2, 3, 2)
        
        for name, traj in trajectories.items():
            color = traj['parameters']['color']
            magnitude = traj['magnitude']
            phase = traj['phase']
            
            ax2.scatter(phase, magnitude, color=color, alpha=0.6, s=20, label=name.replace('_', ' '))
        
        ax2.set_xlabel('Phase (radians)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Magnitude vs Phase')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # الرسم الثالث: تطور المقدار
        ax3 = plt.subplot(2, 3, 3)
        
        for name, traj in trajectories.items():
            color = traj['parameters']['color']
            x = traj['x']
            magnitude = traj['magnitude']
            
            ax3.plot(x, magnitude, color=color, linewidth=2, label=name.replace('_', ' '))
        
        ax3.set_xlabel('x')
        ax3.set_ylabel('Magnitude')
        ax3.set_title('Magnitude Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # الرسم الرابع: تطور الطور
        ax4 = plt.subplot(2, 3, 4)
        
        for name, traj in trajectories.items():
            color = traj['parameters']['color']
            x = traj['x']
            phase = traj['phase']
            
            ax4.plot(x, phase, color=color, linewidth=2, label=name.replace('_', ' '))
        
        ax4.set_xlabel('x')
        ax4.set_ylabel('Phase (radians)')
        ax4.set_title('Phase Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # الرسم الخامس: الانحناء
        ax5 = plt.subplot(2, 3, 5)
        
        for name, traj in trajectories.items():
            color = traj['parameters']['color']
            curvature = traj['curvature']
            
            if len(curvature) > 0:
                ax5.hist(curvature, bins=30, alpha=0.6, color=color, 
                        label=f"{name.replace('_', ' ')} (avg: {np.mean(curvature):.3f})")
        
        ax5.set_xlabel('Curvature')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Curvature Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # الرسم السادس: إحصائيات متقدمة
        ax6 = plt.subplot(2, 3, 6)
        
        # جمع الإحصائيات
        stats_data = []
        labels = []
        
        for name, traj in trajectories.items():
            stats_data.append([
                traj['path_length'],
                abs(traj['winding_number']),
                traj['fractal_dimension'],
                len(traj['critical_points'])
            ])
            labels.append(name.replace('_', ' '))
        
        stats_data = np.array(stats_data)
        
        # رسم بياني شعاعي
        categories = ['Path Length', 'Winding Number', 'Fractal Dim', 'Critical Points']
        
        # تطبيع البيانات
        for i in range(stats_data.shape[1]):
            col_max = np.max(stats_data[:, i])
            if col_max > 0:
                stats_data[:, i] = stats_data[:, i] / col_max
        
        x_pos = np.arange(len(categories))
        width = 0.15
        
        for i, (data, label) in enumerate(zip(stats_data, labels)):
            ax6.bar(x_pos + i * width, data, width, label=label)
        
        ax6.set_xlabel('Metrics')
        ax6.set_ylabel('Normalized Values')
        ax6.set_title('Comparative Statistics')
        ax6.set_xticks(x_pos + width * (len(labels) - 1) / 2)
        ax6.set_xticklabels(categories, rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # حفظ الرسم
        visualization_path = '/home/ubuntu/complex_plane_analysis.png'
        plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ تم حفظ التصور في: {visualization_path}")
        
        return visualization_path
    
    def create_3d_visualization(self):
        """إنشاء تصور ثلاثي الأبعاد"""
        print("\n🌟 إنشاء تصور ثلاثي الأبعاد...")
        
        if 'trajectories' not in self.analysis_results:
            return None
        
        fig = plt.figure(figsize=(15, 10))
        
        # الرسم الأول: مسار ثلاثي الأبعاد (حقيقي، تخيلي، مقدار)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        
        trajectories = self.analysis_results['trajectories']
        
        for name, traj in trajectories.items():
            color = traj['parameters']['color']
            real_part = traj['real_part']
            imag_part = traj['imaginary_part']
            magnitude = traj['magnitude']
            
            ax1.plot(real_part, imag_part, magnitude, color=color, linewidth=2, 
                    label=name.replace('_', ' '), alpha=0.8)
        
        ax1.set_xlabel('Real Part')
        ax1.set_ylabel('Imaginary Part')
        ax1.set_zlabel('Magnitude')
        ax1.set_title('3D Complex Trajectory')
        ax1.legend()
        
        # الرسم الثاني: سطح المقدار
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        
        # اختيار مسار واحد للتفصيل
        main_traj = list(trajectories.values())[0]
        x = main_traj['x']
        real_part = main_traj['real_part']
        imag_part = main_traj['imaginary_part']
        magnitude = main_traj['magnitude']
        
        # إنشاء شبكة
        xi = np.linspace(np.min(real_part), np.max(real_part), 50)
        yi = np.linspace(np.min(imag_part), np.max(imag_part), 50)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # استيفاء المقدار على الشبكة
        try:
            Zi = griddata((real_part, imag_part), magnitude, (Xi, Yi), method='cubic')
            Zi = np.nan_to_num(Zi)
            
            surf = ax2.plot_surface(Xi, Yi, Zi, cmap='viridis', alpha=0.7)
            ax2.set_xlabel('Real Part')
            ax2.set_ylabel('Imaginary Part')
            ax2.set_zlabel('Magnitude')
            ax2.set_title('Magnitude Surface')
            
        except Exception as e:
            print(f"تحذير: لم يتم إنشاء السطح - {e}")
        
        # الرسم الثالث: مسار الطور
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        
        for name, traj in trajectories.items():
            color = traj['parameters']['color']
            x = traj['x']
            phase = traj['phase']
            magnitude = traj['magnitude']
            
            ax3.plot(x, phase, magnitude, color=color, linewidth=2, 
                    label=name.replace('_', ' '), alpha=0.8)
        
        ax3.set_xlabel('x')
        ax3.set_ylabel('Phase')
        ax3.set_zlabel('Magnitude')
        ax3.set_title('Phase-Magnitude Evolution')
        ax3.legend()
        
        # الرسم الرابع: الانحناء والالتواء
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        
        for name, traj in trajectories.items():
            color = traj['parameters']['color']
            curvature = traj['curvature']
            torsion = traj['torsion']
            
            if len(curvature) > 0 and len(torsion) > 0:
                min_len = min(len(curvature), len(torsion))
                x_indices = np.arange(min_len)
                
                ax4.scatter(x_indices, curvature[:min_len], torsion[:min_len], 
                           color=color, alpha=0.6, s=20, label=name.replace('_', ' '))
        
        ax4.set_xlabel('Position Index')
        ax4.set_ylabel('Curvature')
        ax4.set_zlabel('Torsion')
        ax4.set_title('Curvature-Torsion Space')
        ax4.legend()
        
        plt.tight_layout()
        
        # حفظ الرسم
        visualization_3d_path = '/home/ubuntu/complex_3d_analysis.png'
        plt.savefig(visualization_3d_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ تم حفظ التصور ثلاثي الأبعاد في: {visualization_3d_path}")
        
        return visualization_3d_path
    
    def generate_geometric_analysis_report(self):
        """إنشاء تقرير التحليل الهندسي"""
        print("\n📊 إنشاء تقرير التحليل الهندسي...")
        
        if 'trajectories' not in self.analysis_results:
            return None
        
        trajectories = self.analysis_results['trajectories']
        
        report = {
            'analysis_type': 'complex_plane_geometry',
            'trajectories_analyzed': len(trajectories),
            'geometric_insights': [],
            'mathematical_discoveries': [],
            'prime_connections': [],
            'zeta_relationships': []
        }
        
        # تحليل كل مسار
        for name, traj in trajectories.items():
            trajectory_analysis = {
                'name': name,
                'parameters': traj['parameters'],
                'geometric_properties': {
                    'path_length': traj['path_length'],
                    'winding_number': traj['winding_number'],
                    'fractal_dimension': traj['fractal_dimension'],
                    'critical_points_count': len(traj['critical_points']),
                    'inflection_points_count': len(traj['inflection_points']),
                    'spiral_centers_count': len(traj['spiral_centers'])
                },
                'statistical_properties': {
                    'mean_curvature': np.mean(traj['curvature']) if len(traj['curvature']) > 0 else 0,
                    'max_curvature': np.max(traj['curvature']) if len(traj['curvature']) > 0 else 0,
                    'curvature_variance': np.var(traj['curvature']) if len(traj['curvature']) > 0 else 0,
                    'mean_torsion': np.mean(traj['torsion']) if len(traj['torsion']) > 0 else 0
                },
                'periodicity_analysis': traj['periodicity'],
                'symmetry_analysis': traj['symmetries']
            }
            
            # تحليل الاكتشافات
            if traj['fractal_dimension'] > 1.5:
                report['geometric_insights'].append(f"مسار {name} يظهر خصائص فراكتالية قوية")
            
            if abs(traj['winding_number']) > 0.5:
                report['geometric_insights'].append(f"مسار {name} يلف حول الأصل {traj['winding_number']:.2f} مرة")
            
            if len(traj['spiral_centers']) > 0:
                report['geometric_insights'].append(f"مسار {name} يحتوي على {len(traj['spiral_centers'])} مركز حلزوني")
            
            # ربط بالأعداد الأولية وزيتا
            if 'زيتا' in name:
                report['zeta_relationships'].append({
                    'zeta_zero': traj['parameters']['beta'],
                    'geometric_signature': {
                        'path_length': traj['path_length'],
                        'winding_number': traj['winding_number'],
                        'critical_points': len(traj['critical_points'])
                    }
                })
            
            if 'π' in name:
                report['mathematical_discoveries'].append({
                    'pi_connection': True,
                    'periodicity_strength': traj['periodicity']['strength'],
                    'geometric_complexity': traj['fractal_dimension']
                })
        
        # تحليل مقارن
        path_lengths = [traj['path_length'] for traj in trajectories.values()]
        winding_numbers = [abs(traj['winding_number']) for traj in trajectories.values()]
        fractal_dims = [traj['fractal_dimension'] for traj in trajectories.values()]
        
        report['comparative_analysis'] = {
            'path_length_range': [np.min(path_lengths), np.max(path_lengths)],
            'winding_number_range': [np.min(winding_numbers), np.max(winding_numbers)],
            'fractal_dimension_range': [np.min(fractal_dims), np.max(fractal_dims)],
            'most_complex_trajectory': list(trajectories.keys())[np.argmax(fractal_dims)],
            'longest_path_trajectory': list(trajectories.keys())[np.argmax(path_lengths)],
            'highest_winding_trajectory': list(trajectories.keys())[np.argmax(winding_numbers)]
        }
        
        # تقييم الأهمية الثورية
        revolutionary_score = 0
        revolutionary_score += len(report['geometric_insights']) * 2
        revolutionary_score += len(report['zeta_relationships']) * 5
        revolutionary_score += len(report['mathematical_discoveries']) * 3
        
        if revolutionary_score >= 20:
            report['revolutionary_potential'] = 'عالي جداً'
        elif revolutionary_score >= 10:
            report['revolutionary_potential'] = 'عالي'
        elif revolutionary_score >= 5:
            report['revolutionary_potential'] = 'متوسط'
        else:
            report['revolutionary_potential'] = 'منخفض'
        
        print(f"✅ تم إنشاء التقرير - الإمكانية الثورية: {report['revolutionary_potential']}")
        print(f"✅ اكتشافات هندسية: {len(report['geometric_insights'])}")
        print(f"✅ علاقات زيتا: {len(report['zeta_relationships'])}")
        print(f"✅ اكتشافات رياضية: {len(report['mathematical_discoveries'])}")
        
        return report

def main():
    """الدالة الرئيسية لتحليل المستوى المركب"""
    print("🧮 محلل السلوك في المستوى المركب")
    print("تطوير: باسل يحيى عبدالله")
    print("المرحلة الثانية: التحليل الهندسي المتقدم")
    print("=" * 60)
    
    # إنشاء المحلل
    analyzer = ComplexPlaneAnalyzer()
    
    # المرحلة 1: تحليل المسارات المركبة
    trajectories = analyzer.analyze_complex_trajectories()
    
    # المرحلة 2: إنشاء التصورات البصرية
    viz_2d = analyzer.create_complex_plane_visualization()
    viz_3d = analyzer.create_3d_visualization()
    
    # المرحلة 3: إنشاء التقرير الهندسي
    geometric_report = analyzer.generate_geometric_analysis_report()
    
    print("\n" + "=" * 60)
    print("🎉 اكتمل التحليل الهندسي!")
    print(f"🌟 الإمكانية الثورية: {geometric_report['revolutionary_potential']}")
    print(f"🔍 مسارات محللة: {geometric_report['trajectories_analyzed']}")
    print(f"📊 اكتشافات هندسية: {len(geometric_report['geometric_insights'])}")
    print(f"🧮 علاقات زيتا: {len(geometric_report['zeta_relationships'])}")
    
    return analyzer, geometric_report, viz_2d, viz_3d

if __name__ == "__main__":
    analyzer, report, viz_2d, viz_3d = main()

