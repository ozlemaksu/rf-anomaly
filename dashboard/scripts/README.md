# dashboard/scripts

Dashboard verilerini (`dashboard/data/`) hazırlayan ve sonuçları kontrol eden betikler (C tarafı).
Bunlar geliştirme sırasındaki yerel proje düzeninden taşındı: içlerindeki `data/...` yolları
o düzene göredir ve başka bir ortamda çalıştırmak için uyarlanması gerekebilir.
Dashboard'ın kendisi (`../demo.py`) bu betiklere ihtiyaç duymaz.

Not: `energy_baseline.py` baseline'ı bağımsız olarak yeniden çalıştırır (eşik 1.2532, doğruluk 0.775).
Resmi sonuç `ml/results/v2` içindeki kayıttır (eşik 1.2356, doğruluk 0.783); ROC-AUC ikisinde de 0.7568.
