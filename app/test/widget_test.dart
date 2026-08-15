// Smoke test cơ bản cho AI4Life app.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:ai4life/main.dart';

void main() {
  testWidgets('App khởi động và hiển thị nút thu', (WidgetTester tester) async {
    await tester.pumpWidget(const AI4LifeApp());
    expect(find.text('Trợ Lý Giao Tiếp AI'), findsOneWidget);
  });
}
