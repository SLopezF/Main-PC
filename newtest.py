import motor_lib, encoder_lib
e = encoder_lib.Encoder(); m = motor_lib.Motor()
m.deshabilitar(); m.set_corriente(600); m.set_micropasos(8); m.habilitar(); m.zero()
a0 = e.leer_angulo()[1]; m.ir_a_grados(90); m.esperar_fin()
a1 = e.leer_angulo()[1]
print("motor 90 deg -> encoder", a1 - a0, "deg de eje")
