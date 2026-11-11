import time

from cps_client import CPSClient


def motor_ft_data(ip="192.168.10.10", port=10003, read_hz=30):
    cps = CPSClient()
    ret = cps.HRIF_Connect(0, ip, port)
    if ret != 0:
        print(f"Connection error, with code {ret}")
        return
    print("Succesfully connect! Now read the data!")
    dt = 1.0 / read_hz
    try:
        while True:
            start_time = time.time()
            result = []
            ret = cps.HRIF_ReadFTCabData(0, 0, result)
            if ret == 0 and len(result) >= 6:
                fx = float(result[0])
                fy = float(result[1])
                fz = float(result[2])
                tx = float(result[3])
                ty = float(result[4])
                tz = float(result[5])
                # print("read {}")
                print(
                    f"{time.strftime('%H:%M:%S'):<10}FTCab | {fx:>8.2f} {fy:>8.2f}"
                    f" {fz:>8.2f} | {tx:>8.3f} {ty:>8.3f} {tz:>8.3f}"
                )
            else:
                print(f"ReadFTCabData error with ret {ret}")

            ret = cps.HRIF_ReadForceData(0, 0, result)
            
            if ret == 0 and len(result) >= 6:
                fx = float(result[0])
                fy = float(result[1])
                fz = float(result[2])
                tx = float(result[3])
                ty = float(result[4])
                tz = float(result[5])
                # print("read {}")
                print(
                    f"{time.strftime('%H:%M:%S'):<10}Force | {fx:>8.2f} {fy:>8.2f}"
                    f" {fz:>8.2f} | {tx:>8.3f} {ty:>8.3f} {tz:>8.3f}"
                )
            else:
                print(f"HRIF_ReadForceData error with ret {ret}")
            print("-----------------------------------------------------------------")
            
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("Enter Ctrl+C, exit...")
    finally:
        cps.HRIF_DisConnect(0)
        print("successfully disconnect")


if __name__ == "__main__":
    # print("hello world")
    motor_ft_data()