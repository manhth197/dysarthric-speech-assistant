import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:record/record.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() => runApp(const AI4LifeApp());

class AI4LifeApp extends StatelessWidget {
  const AI4LifeApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI4Life',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        colorSchemeSeed: const Color(0xFF3b82f6),
        scaffoldBackgroundColor: const Color(0xFF0f172a),
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _player = AudioPlayer();

  bool _recording = false;
  bool _busy = false;
  String _status = 'Nhấn nút để bắt đầu nói';
  String _asr = '';
  String _gec = '';
  String? _lastAudio;
  String _serverUrl = 'http://192.168.0.74:8000';
  String _authToken = '';

  @override
  void initState() {
    super.initState();
    _loadPrefs();
    // Khi phát xong audio -> trả trạng thái về sẵn sàng
    _player.onPlayerComplete.listen((_) {
      if (mounted && !_recording && !_busy) {
        setState(() => _status = 'Nhấn nút để nói tiếp');
      }
    });
  }

  Future<void> _loadPrefs() async {
    final p = await SharedPreferences.getInstance();
    setState(() {
      _serverUrl = p.getString('server_url') ?? _serverUrl;
      _authToken = p.getString('auth_token') ?? '';
    });
  }

  Future<void> _savePrefs(String url, String token) async {
    final p = await SharedPreferences.getInstance();
    await p.setString('server_url', url);
    await p.setString('auth_token', token);
    setState(() {
      _serverUrl = url;
      _authToken = token;
    });
  }

  Future<void> _onMainButton() async {
    if (_busy) return;
    if (_recording) {
      await _stopAndProcess();
    } else {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
    if (!await _recorder.hasPermission()) {
      setState(() => _status = '⚠️ Cần cấp quyền micro');
      return;
    }
    final dir = await getTemporaryDirectory();
    final path = '${dir.path}/record.wav';
    await _recorder.start(
      const RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
      ),
      path: path,
    );
    HapticFeedback.mediumImpact(); // rung xác nhận bắt đầu ghi
    setState(() {
      _recording = true;
      _asr = '';
      _gec = '';
      _status = '🔴 Đang nghe... nhấn lại để DỪNG';
    });
  }

  Future<void> _stopAndProcess() async {
    final path = await _recorder.stop();
    HapticFeedback.mediumImpact(); // rung xác nhận đã dừng ghi
    setState(() {
      _recording = false;
      _busy = true;
      _status = '⏳ Đang gửi & xử lý...';
    });
    if (path == null) {
      setState(() {
        _busy = false;
        _status = '❌ Không ghi được âm thanh';
      });
      return;
    }
    try {
      final uri = Uri.parse('$_serverUrl/process');
      final req = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('audio', path));
      if (_authToken.isNotEmpty) {
        req.headers['X-Auth-Token'] = _authToken; // khớp AI4LIFE_TOKEN ở server
      }
      final streamed = await req.send().timeout(const Duration(seconds: 60));
      final body = await streamed.stream.bytesToString();
      final data = jsonDecode(body) as Map<String, dynamic>;
      if (streamed.statusCode == 200) {
        setState(() {
          _asr = (data['asr_text'] ?? '').toString();
          _gec = (data['gec_text'] ?? '').toString();
          _status = '✅ Xong (${data['process_time']}s)';
        });
        final b64 = data['audio_base64'];
        if (b64 is String && b64.isNotEmpty) {
          await _playBase64(b64);
        }
      } else {
        setState(() =>
            _status = '❌ Lỗi: ${data['error'] ?? streamed.statusCode}');
      }
    } catch (e) {
      setState(() => _status =
          '❌ Không kết nối được máy chủ.\nKiểm tra WiFi và địa chỉ trong ⚙️.');
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _playBase64(String b64) async {
    final dir = await getTemporaryDirectory();
    final f = File('${dir.path}/output.wav');
    await f.writeAsBytes(base64Decode(b64));
    _lastAudio = f.path;
    if (mounted) setState(() => _status = '🔊 Đang phát...');
    await _player.play(DeviceFileSource(f.path));
  }

  Future<void> _replay() async {
    if (_lastAudio != null) {
      if (mounted) setState(() => _status = '🔊 Đang phát...');
      await _player.play(DeviceFileSource(_lastAudio!));
    }
  }

  Future<void> _openSettings() async {
    final urlCtl = TextEditingController(text: _serverUrl);
    final tokenCtl = TextEditingController(text: _authToken);
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Cài đặt máy chủ'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: urlCtl,
              keyboardType: TextInputType.url,
              decoration: const InputDecoration(
                labelText: 'Địa chỉ máy chủ (PC)',
                hintText: 'http://192.168.0.74:8000',
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: tokenCtl,
              decoration: const InputDecoration(
                labelText: 'Mã bảo mật (tùy chọn)',
                hintText: 'khớp AI4LIFE_TOKEN trên PC',
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Hủy'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Lưu'),
          ),
        ],
      ),
    );
    if (ok == true) {
      final url = urlCtl.text.trim();
      if (url.isNotEmpty) {
        await _savePrefs(url, tokenCtl.text.trim());
      }
    }
  }

  @override
  void dispose() {
    _recorder.dispose();
    _player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final Color btnColor = _busy
        ? Colors.grey
        : (_recording ? const Color(0xFFef4444) : const Color(0xFF3b82f6));
    return Scaffold(
      appBar: AppBar(
        title: const Text('Trợ Lý Giao Tiếp AI'),
        centerTitle: true,
        actions: [
          IconButton(
            tooltip: 'Cài đặt máy chủ',
            icon: const Icon(Icons.settings),
            onPressed: _openSettings,
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              Text(
                _status,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 20, height: 1.4),
              ),
              const SizedBox(height: 28),
              GestureDetector(
                onTap: _onMainButton,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  width: 220,
                  height: 220,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: btnColor,
                    boxShadow: [
                      BoxShadow(
                        color: btnColor.withValues(alpha: 0.5),
                        blurRadius: 30,
                        spreadRadius: 4,
                      ),
                    ],
                  ),
                  child: _busy
                      ? const Center(
                          child: SizedBox(
                            width: 70,
                            height: 70,
                            child: CircularProgressIndicator(
                              color: Colors.white,
                              strokeWidth: 6,
                            ),
                          ),
                        )
                      : Icon(
                          _recording ? Icons.stop_rounded : Icons.mic_rounded,
                          size: 110,
                          color: Colors.white,
                        ),
                ),
              ),
              const SizedBox(height: 28),
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      if (_asr.isNotEmpty)
                        _resultCard('🔍 Nghe được', _asr, fontSize: 20),
                      if (_gec.isNotEmpty)
                        _resultCard('✨ Câu chuẩn', _gec,
                            fontSize: 30, bold: true, highlight: true),
                    ],
                  ),
                ),
              ),
              if (_gec.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: FilledButton.icon(
                    onPressed: _replay,
                    icon: const Icon(Icons.volume_up, size: 28),
                    label:
                        const Text('Phát lại', style: TextStyle(fontSize: 22)),
                    style: FilledButton.styleFrom(
                      minimumSize: const Size.fromHeight(60),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _resultCard(String title, String text,
      {double fontSize = 20, bool bold = false, bool highlight = false}) {
    return Card(
      color: highlight ? const Color(0xFF1e3a8a) : const Color(0xFF1e293b),
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title,
                style: const TextStyle(fontSize: 14, color: Colors.white70)),
            const SizedBox(height: 8),
            Text(
              text,
              style: TextStyle(
                fontSize: fontSize,
                fontWeight: bold ? FontWeight.bold : FontWeight.normal,
                height: 1.35,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
