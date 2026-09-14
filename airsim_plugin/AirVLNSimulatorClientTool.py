import msgpackrpc
import time
import airsim
import threading
import random
import copy
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == '__main__':
    import sys
    cur_path=os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, cur_path+"/..")
    print(os.getcwd())
    from src.common.param import args
else:
    from src.common.param import args

from utils.logger import logger


class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.flag_ok = False

    def run(self):
        try:
            self.result = self.func(*self.args)
        except Exception as e:
            logger.error(e)
            self.flag_ok = False
        else:
            self.flag_ok = True

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except:
            return None


class AirVLNSimulatorClientTool:
    def __init__(self, machines_info) -> None:
        self.machines_info = copy.deepcopy(machines_info)
        self.socket_clients = []
        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in machines_info ]
        preview_dir = os.environ.get("AIRVLN_LIVE_PREVIEW_DIR", "").strip()
        self.live_preview_dir = Path(preview_dir) if preview_dir else None
        self.live_preview_camera = os.environ.get("AIRVLN_LIVE_PREVIEW_CAMERA", "preview_0").strip() or "preview_0"
        self.live_preview_steps = max(1, int(os.environ.get("AIRVLN_LIVE_POSE_TWEEN_STEPS", "1")))
        self.live_preview_max_tween_distance = max(
            20.0,
            float(os.environ.get("AIRVLN_LIVE_POSE_TWEEN_MAX_DISTANCE", "120.0")),
        )
        self.live_preview_meters_per_tween = max(
            1.0,
            float(os.environ.get("AIRVLN_LIVE_POSE_TWEEN_METERS_PER_STEP", "4.0")),
        )
        self.live_preview_max_tween_steps = max(
            self.live_preview_steps,
            int(os.environ.get("AIRVLN_LIVE_POSE_TWEEN_MAX_STEPS", "36")),
        )
        self.live_preview_quality = min(95, max(45, int(os.environ.get("AIRVLN_LIVE_JPEG_QUALITY", "90"))))
        self.live_preview_max_width = max(240, int(os.environ.get("AIRVLN_LIVE_PREVIEW_MAX_WIDTH", "768")))
        # Tween poses are used for collision-safe motion, but intermediate
        # tween screenshots are redundant and make AirSim image RPC the
        # dominant cost. Capture one display frame at the final pose by default.
        self.live_preview_capture_every_tween = os.environ.get(
            "AIRVLN_LIVE_PREVIEW_CAPTURE_EVERY_TWEEN", "0"
        ).lower() not in ("0", "false", "no")
        # The mandatory observation RPC can include preview_0.  This avoids a
        # second high-resolution image RPC after every model action.
        self.live_preview_capture_in_observation = os.environ.get(
            "AIRVLN_LIVE_PREVIEW_CAPTURE_IN_OBSERVATION", "0"
        ).lower() not in ("0", "false", "no")
        self.live_low_light_enhance = os.environ.get("AIRVLN_LIVE_LOW_LIGHT_ENHANCE", "0").lower() not in ("0", "false", "no")
        self.live_preview_depth_guard = os.environ.get("AIRVLN_LIVE_PREVIEW_DEPTH_GUARD", "0").lower() not in ("0", "false", "no")
        self.live_fast_pose_verify = os.environ.get(
            "AIRVLN_LIVE_FAST_POSE_VERIFY", "0"
        ).lower() not in ("0", "false", "no")
        self.live_fast_pose_verify_interval = max(
            1, int(os.environ.get("AIRVLN_LIVE_FAST_POSE_VERIFY_INTERVAL", "8"))
        )
        self._live_preview_index = 0
        self._live_preview_lock = threading.Lock()
        self._live_preview_failures = 0
        self._live_pose_initialized = False
        self._live_pose_verify_index = 0
        self._collision_lock = threading.Lock()
        self.last_collision_event = None
        if self.live_preview_dir is not None:
            self.live_preview_dir.mkdir(parents=True, exist_ok=True)

        self._init_check()

    def _init_check(self) -> None:
        ips = [item['MACHINE_IP'] for item in self.machines_info]
        assert len(ips) == len(set(ips)), 'MACHINE_IP repeat'

    def _confirmSocketConnection(self, socket_client: msgpackrpc.Client) -> bool:
        try:
            socket_client.call('ping')
            logger.info("Connected\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            return True
        except:
            try:
                logger.error("Ping returned false\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            except:
                logger.error('Ping returned false')
            return False

    def _waitForRpcConnection(self, ip: str, port: int) -> None:
        retry_count = max(
            1, int(os.environ.get("AIRVLN_CONFIRM_RETRY_COUNT", "12"))
        )
        retry_sleep = max(
            0.1, float(os.environ.get("AIRVLN_CONFIRM_RETRY_SLEEP", "5"))
        )
        last_error = None
        for attempt in range(retry_count):
            probe = msgpackrpc.Client(
                msgpackrpc.Address(ip, int(port)), timeout=10
            )
            try:
                probe.call("ping")
                logger.info(
                    "AirSim RPC ready at %s:%s on attempt %d/%d",
                    ip,
                    port,
                    attempt + 1,
                    retry_count,
                )
                return
            except Exception as error:
                last_error = error
                logger.warning(
                    "AirSim RPC wait %s:%s attempt %d/%d failed: %s",
                    ip,
                    port,
                    attempt + 1,
                    retry_count,
                    error,
                )
                if attempt < retry_count - 1:
                    time.sleep(retry_sleep)
            finally:
                try:
                    probe.close()
                except Exception:
                    pass
        if last_error is not None:
            raise last_error

    def _confirmConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    self.airsim_clients[index_1][index_2].confirmConnection()

        return

    def _confirmConnection_with_retry(self, retry_count: int = 12, retry_sleep: float = 5.0) -> None:
        retry_count = max(
            retry_count,
            int(os.environ.get("AIRVLN_CONFIRM_RETRY_COUNT", retry_count)),
        )
        retry_sleep = max(
            retry_sleep,
            float(os.environ.get("AIRVLN_CONFIRM_RETRY_SLEEP", retry_sleep)),
        )
        last_error = None
        for attempt in range(retry_count):
            try:
                self._confirmConnection()
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    "confirmConnection retry %d/%d failed: %s",
                    attempt + 1,
                    retry_count,
                    e,
                )
                if attempt < retry_count - 1:
                    time.sleep(retry_sleep)

        if last_error is not None:
            raise last_error

    def _probe_image_stream_once(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, airsim_client in enumerate(self.airsim_clients[index_1]):
                if airsim_client is None:
                    continue

                responses = airsim_client.simGetImages(
                    [
                        airsim.ImageRequest(
                            "front_0",
                            airsim.ImageType.Scene,
                            pixels_as_float=False,
                            compress=False,
                        )
                    ],
                    vehicle_name='Drone_1',
                )
                assert len(responses) == 1, 'Failed to retrieve warmup image'
                response_rgb = responses[0]
                assert response_rgb.height == args.Image_Height_RGB, 'Warmup RGB height mismatch'
                assert response_rgb.width == args.Image_Width_RGB, 'Warmup RGB width mismatch'
                img1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
                assert img1d.size == response_rgb.height * response_rgb.width * 3, 'Warmup RGB payload mismatch'

    def wait_for_image_streams(self, retry_count: int = 6, retry_sleep: float = 5.0, initial_sleep: float = 5.0) -> None:
        if initial_sleep > 0:
            time.sleep(initial_sleep)

        last_error = None
        for attempt in range(retry_count):
            try:
                self._probe_image_stream_once()
                logger.info(
                    "Image stream warmup succeeded on attempt %d/%d",
                    attempt + 1,
                    retry_count,
                )
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    "Image stream warmup retry %d/%d failed: %s",
                    attempt + 1,
                    retry_count,
                    e,
                )
                if attempt < retry_count - 1:
                    time.sleep(retry_sleep)

        if last_error is not None:
            raise last_error

    def _closeSocketConnection(self) -> None:
        socket_clients = self.socket_clients

        for socket_client in socket_clients:
            try:
                socket_client.close()
            except Exception as e:
                pass

        self.socket_clients = []
        return

    def _closeConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    try:
                        self.airsim_clients[index_1][index_2].close()
                    except Exception as e:
                        pass

        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in self.machines_info]
        return

    def run_call(self, airsim_timeout: int=15) -> None:
        socket_clients = []
        for index, item in enumerate(self.machines_info):
            socket_clients.append(
                msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=180)
            )

        for socket_client in socket_clients:
            if not self._confirmSocketConnection(socket_client):
                logger.error('cannot establish socket')
                raise Exception('cannot establish socket')

        self.socket_clients = socket_clients


        before = time.time()
        self._closeConnection()

        def _run_command(index, socket_client: msgpackrpc.Client):
            logger.info(f'Failed to open scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}')
            result = socket_client.call('reopen_scenes', socket_client.address._host, self.machines_info[index]['open_scenes'])

            if result[0] == False:
                logger.error(f'Failed to open scenes, machine : {socket_client.address._host}:{socket_client.address._port}')
                raise Exception('Failed to open scenes')
            assert len(result[1]) == 2, 'Failed to open scenes'

            ip = result[1][0]
            ports = result[1][1]

            if isinstance(ip, bytes):
                ip = ip.decode()

            assert str(ip) == str(socket_client.address._host), 'Failed to open scenes'
            assert len(ports) == len(self.machines_info[index]['open_scenes']), 'Failed to open scenes'
            for i, port in enumerate(ports):
                if self.machines_info[index]['open_scenes'][i] is None:
                    self.airsim_clients[index][i] = None
                else:
                    self._waitForRpcConnection(ip, int(port))
                    self.airsim_clients[index][i] = airsim.VehicleClient(ip=ip, port=port, timeout_value=airsim_timeout)

            logger.info(f'Failed to open scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}')
            return

        threads = []
        thread_results = []
        for index, socket_client in enumerate(socket_clients):
            threads.append(
                MyThread(_run_command, (index, socket_client))
            )
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            thread.get_result()
            thread_results.append(thread.flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            raise Exception('Failed to open scenes')

        after = time.time()
        diff = after - before
        logger.info(f"Start time: {diff}")

        time.sleep(2.0)
        self._confirmConnection_with_retry()
        self._closeSocketConnection()

    def attach_existing_scenes(self, airsim_timeout: int = 15) -> None:
        """Attach to a controlled, already-loaded AirSim scene.

        Presentation batches keep a single Scene 16 instance alive.  Reopening
        it per episode costs roughly twenty seconds and does not improve
        episode isolation because the environment still applies each episode's
        initial pose during reset.  This method is deliberately opt-in via
        the environment layer and assumes the caller owns the simulator lock.
        """
        self._closeConnection()
        for machine_index, item in enumerate(self.machines_info):
            ip = str(item["MACHINE_IP"])
            first_scene_port = int(item["SOCKET_PORT"]) + 1
            for scene_index, scene_id in enumerate(item["open_scenes"]):
                if scene_id is None:
                    self.airsim_clients[machine_index][scene_index] = None
                    continue
                port = first_scene_port + scene_index
                self._waitForRpcConnection(ip, port)
                self.airsim_clients[machine_index][scene_index] = airsim.VehicleClient(
                    ip=ip,
                    port=port,
                    timeout_value=airsim_timeout,
                )
        self._confirmConnection_with_retry(retry_count=3, retry_sleep=0.5)

    def _save_live_preview_response(self, response) -> None:
        """Persist a preview_0 image supplied by the normal observation RPC."""
        if self.live_preview_dir is None or self._live_preview_failures >= 3:
            return
        try:
            height = int(getattr(response, "height", 0) or 0)
            width = int(getattr(response, "width", 0) or 0)
            if height <= 0 or width <= 0:
                return
            raw = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            if raw.size != height * width * 3:
                return
            rgb = raw.reshape(height, width, 3)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            mean_luma = float(gray.mean())
            std_luma = float(gray.std())
            black_ratio = float((gray < 8).sum()) / float(gray.size)
            if mean_luma < 3.0 or (mean_luma < 10.0 and std_luma < 2.0) or black_ratio > 0.92:
                logger.warning(
                    "Skipped invalid observation preview frame: mean=%.2f std=%.2f black_ratio=%.3f",
                    mean_luma,
                    std_luma,
                    black_ratio,
                )
                return
            if self.live_low_light_enhance and mean_luma < 48.0:
                alpha = min(2.25, max(1.15, 58.0 / max(mean_luma, 1.0)))
                rgb = cv2.convertScaleAbs(rgb, alpha=alpha, beta=3.0)
            if width > self.live_preview_max_width:
                scale = float(self.live_preview_max_width) / float(width)
                rgb = cv2.resize(
                    rgb,
                    (self.live_preview_max_width, max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            ok, jpeg = cv2.imencode(
                ".jpg",
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), self.live_preview_quality],
            )
            if not ok:
                return
            with self._live_preview_lock:
                index = self._live_preview_index
                self._live_preview_index += 1
            final_path = self.live_preview_dir / f"preview_{index:08d}.jpg"
            temp_path = self.live_preview_dir / f"preview_{index:08d}.tmp"
            with temp_path.open("wb") as fp:
                fp.write(jpeg.tobytes())
            os.replace(str(temp_path), str(final_path))
        except Exception as exc:
            self._live_preview_failures += 1
            logger.warning(
                "Observation preview capture failed (%d/3): %s",
                self._live_preview_failures,
                exc,
            )

    def getImageResponses(self, get_rgb=True, get_depth=True):

        def _getImages(airsim_client: airsim.VehicleClient, scen_id, get_rgb, get_depth):
            if airsim_client is None:
                raise Exception('error')
                return None, None

            img_rgb = None
            img_depth = None
            capture_preview = bool(
                self.live_preview_dir is not None
                and self.live_preview_capture_in_observation
                and get_rgb
            )

            if not get_rgb and not get_depth:
                return None, None

            if scen_id in [1, 7]:
                time_sleep_cnt = 0
                while True:
                    try:
                        ImageRequest = []
                        if get_rgb:
                            ImageRequest.append(
                                airsim.ImageRequest("front_0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
                            )
                        if get_depth:
                            ImageRequest.append(
                                airsim.ImageRequest("front_0", airsim.ImageType.DepthVis, pixels_as_float=False, compress=True)
                            )
                        if capture_preview:
                            ImageRequest.append(
                                airsim.ImageRequest(
                                    self.live_preview_camera,
                                    airsim.ImageType.Scene,
                                    pixels_as_float=False,
                                    compress=False,
                                )
                            )

                        responses = airsim_client.simGetImages(ImageRequest, vehicle_name='Drone_1')

                        if get_rgb and get_depth:
                            response_rgb = responses[0]
                            response_depth = responses[1]
                        elif get_rgb and not get_depth:
                            response_rgb = responses[0]
                        elif not get_rgb and get_depth:
                            response_depth = responses[0]
                        else:
                            break
                        response_preview = responses[-1] if capture_preview else None


                        img_rgb = None
                        img_depth = None

                        if get_rgb:
                            assert response_rgb.height == args.Image_Height_RGB and response_rgb.width == args.Image_Width_RGB, 'Failed to retrieve RGB image'

                            img1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
                            if args.run_type not in ['eval']:
                                assert not (img1d.flatten()[0] == img1d).all(), 'Failed to retrieve RGB image'
                            img_rgb = img1d.reshape(response_rgb.height, response_rgb.width, 3)
                            img_rgb = np.array(img_rgb)

                        if get_depth:
                            assert response_depth.height == args.Image_Height_DEPTH and response_depth.width == args.Image_Width_DEPTH, 'Failed to retrieve DEPTH image'

                            png_file_name = '/tmp/AirVLN_depth_{}_{}.png'.format(time.time(), random.randint(0, 10000))
                            airsim.write_file(png_file_name, response_depth.image_data_uint8)
                            img3d = cv2.imread(png_file_name)

                            os.remove(png_file_name)

                            img1d = img3d[:, :, 1]
                            img1d = img1d.reshape(response_depth.height, response_depth.width, 1)

                            obs_depth_img = img1d / 255

                            img_depth = np.array(obs_depth_img, dtype=np.float32)

                        if response_preview is not None:
                            self._save_live_preview_response(response_preview)

                        break
                    except:
                        time_sleep_cnt += 1
                        logger.error("Image retrieval error")
                        logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                        time.sleep(1)

                    if time_sleep_cnt > 20:
                        raise Exception('Failed to retrieve image')

            else:
                time_sleep_cnt = 0
                while True:
                    try:
                        ImageRequest = []
                        if get_rgb:
                            ImageRequest.append(
                                airsim.ImageRequest("front_0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
                            )
                        if get_depth:
                            ImageRequest.append(
                                airsim.ImageRequest("front_0", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
                            )
                        if capture_preview:
                            ImageRequest.append(
                                airsim.ImageRequest(
                                    self.live_preview_camera,
                                    airsim.ImageType.Scene,
                                    pixels_as_float=False,
                                    compress=False,
                                )
                            )

                        responses = airsim_client.simGetImages(ImageRequest, vehicle_name='Drone_1')

                        if get_rgb and get_depth:
                            response_rgb = responses[0]
                            response_depth = responses[1]
                        elif get_rgb and not get_depth:
                            response_rgb = responses[0]
                        elif not get_rgb and get_depth:
                            response_depth = responses[0]
                        else:
                            break
                        response_preview = responses[-1] if capture_preview else None

                        if get_rgb:
                            assert response_rgb.height == args.Image_Height_RGB and response_rgb.width == args.Image_Width_RGB, 'Failed to retrieve RGB image'

                            img1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
                            img_rgb = img1d.reshape(response_rgb.height, response_rgb.width, 3)
                            img_rgb = np.array(img_rgb)

                        if get_depth:
                            assert response_depth.height == args.Image_Height_DEPTH and response_depth.width == args.Image_Width_DEPTH, 'Failed to retrieve DEPTH image'

                            depth_img_in_meters = airsim.list_to_2d_float_array(response_depth.image_data_float, response_depth.width, response_depth.height)
                            if depth_img_in_meters.min() < 1e4:
                                assert not (depth_img_in_meters.flatten()[0] == depth_img_in_meters).all(), 'Failed to retrieve DEPTH image'
                            depth_img_in_meters = depth_img_in_meters.reshape(response_depth.height, response_depth.width, 1)

                            obs_depth_img = np.clip(depth_img_in_meters, 0, 100)
                            obs_depth_img = obs_depth_img / 100

                            img_depth = np.array(obs_depth_img, dtype=np.float32)

                        if response_preview is not None:
                            self._save_live_preview_response(response_preview)

                        break
                    except:
                        time_sleep_cnt += 1
                        logger.error("Failed to retrieve image")
                        logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                        time.sleep(1)

                    if time_sleep_cnt > 20:
                        raise Exception('Failed to retrieve image')

            # Tip: If you are using AirVLN code for the first time, please confirm that the
            #       channel order of the images captured is as expected by visualization!
            # Example is as below:

            # plt.imsave('./tmp/img_rgb.png', img_rgb)

            # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

            # plt.imsave('./tmp/img_rgb_converted.png', img_rgb)

            # plt.imsave('./tmp/img_depth.png', img_depth.squeeze(), cmap='gray')

            return img_rgb, img_depth

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_getImages, (self.airsim_clients[index_1][index_2], self.machines_info[index_1]['open_scenes'][index_2], get_rgb, get_depth))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        responses = []
        for index_1, _ in enumerate(threads):
            responses.append([])
            for index_2, _ in enumerate(threads[index_1]):
                responses[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getImageResponses failed')
            return None

        return responses

    def setPoses(self, poses: list) -> bool:
        def _pose_position(pose):
            return [
                float(pose.position.x_val),
                float(pose.position.y_val),
                float(pose.position.z_val),
            ]

        def _sync_pose(target_pose, actual_pose):
            target_pose.position.x_val = float(actual_pose.position.x_val)
            target_pose.position.y_val = float(actual_pose.position.y_val)
            target_pose.position.z_val = float(actual_pose.position.z_val)
            target_pose.orientation.x_val = float(actual_pose.orientation.x_val)
            target_pose.orientation.y_val = float(actual_pose.orientation.y_val)
            target_pose.orientation.z_val = float(actual_pose.orientation.z_val)
            target_pose.orientation.w_val = float(actual_pose.orientation.w_val)

        def _record_collision(reason, collision_info, requested_pose, safe_pose):
            event = {
                "time": time.time(),
                "reason": reason,
                "object_name": str(getattr(collision_info, "object_name", "") or ""),
                "penetration_depth": float(getattr(collision_info, "penetration_depth", 0.0) or 0.0),
                "requested_position": _pose_position(requested_pose),
                "safe_position": _pose_position(safe_pose),
            }
            with self._collision_lock:
                self.last_collision_event = event

        def _interpolate_pose(start_pose, target_pose, alpha):
            pose = copy.deepcopy(target_pose)
            pose.position.x_val = start_pose.position.x_val + (target_pose.position.x_val - start_pose.position.x_val) * alpha
            pose.position.y_val = start_pose.position.y_val + (target_pose.position.y_val - start_pose.position.y_val) * alpha
            pose.position.z_val = start_pose.position.z_val + (target_pose.position.z_val - start_pose.position.z_val) * alpha

            start_q = np.array([
                start_pose.orientation.x_val,
                start_pose.orientation.y_val,
                start_pose.orientation.z_val,
                start_pose.orientation.w_val,
            ], dtype=np.float64)
            target_q = np.array([
                target_pose.orientation.x_val,
                target_pose.orientation.y_val,
                target_pose.orientation.z_val,
                target_pose.orientation.w_val,
            ], dtype=np.float64)
            if np.dot(start_q, target_q) < 0:
                target_q = -target_q
            # Ease camera rotation while keeping translation linear and predictable.
            rotation_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            dot = float(np.clip(np.dot(start_q, target_q), -1.0, 1.0))
            if dot > 0.9995:
                quat = start_q + (target_q - start_q) * rotation_alpha
            else:
                theta = float(np.arccos(dot))
                sin_theta = float(np.sin(theta))
                quat = (
                    np.sin((1.0 - rotation_alpha) * theta) / sin_theta * start_q
                    + np.sin(rotation_alpha * theta) / sin_theta * target_q
                )
            norm = np.linalg.norm(quat)
            if norm > 1e-8:
                quat /= norm
                pose.orientation.x_val = float(quat[0])
                pose.orientation.y_val = float(quat[1])
                pose.orientation.z_val = float(quat[2])
                pose.orientation.w_val = float(quat[3])
            return pose

        def _capture_preview(airsim_client, include_depth=False):
            if self.live_preview_dir is None or self._live_preview_failures >= 3:
                return None
            try:
                camera_names = [self.live_preview_camera]
                if self.live_preview_camera != "front_0":
                    camera_names.append("front_0")
                responses = []
                for camera_name in camera_names:
                    requests = [
                        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
                    ]
                    if include_depth:
                        requests.append(
                            airsim.ImageRequest(
                                "front_0",
                                airsim.ImageType.DepthPerspective,
                                pixels_as_float=True,
                                compress=False,
                            )
                        )
                    responses = airsim_client.simGetImages(
                        requests,
                        vehicle_name="Drone_1",
                    )
                    if responses and responses[0].height > 0 and responses[0].width > 0:
                        break
                if not responses or responses[0].height <= 0 or responses[0].width <= 0:
                    return None
                response = responses[0]
                rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
                depth_guard = None
                if include_depth and len(responses) > 1:
                    depth_response = responses[1]
                    depth = airsim.list_to_2d_float_array(
                        depth_response.image_data_float,
                        depth_response.width,
                        depth_response.height,
                    )
                    depth = np.asarray(depth, dtype=np.float32)
                    h, w = depth.shape[:2]
                    center = depth[h // 3 : h * 2 // 3, w // 3 : w * 2 // 3]
                    valid = center[np.isfinite(center) & (center > 0.01)]
                    if valid.size:
                        depth_guard = {
                            "center_p10_m": float(np.percentile(valid, 10)),
                            "center_p25_m": float(np.percentile(valid, 25)),
                            "center_under_1m_ratio": float((valid < 1.0).sum()) / float(valid.size),
                        }

                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                mean_luma = float(gray.mean())
                std_luma = float(gray.std())
                black_ratio = float((gray < 8).sum()) / float(gray.size)
                if mean_luma < 3.0 or (mean_luma < 10.0 and std_luma < 2.0) or black_ratio > 0.92:
                    logger.warning(
                        "Skipped invalid black preview frame: mean=%.2f std=%.2f black_ratio=%.3f",
                        mean_luma,
                        std_luma,
                        black_ratio,
                    )
                    return depth_guard
                if self.live_low_light_enhance and mean_luma < 48.0:
                    alpha = min(2.25, max(1.15, 58.0 / max(mean_luma, 1.0)))
                    rgb = cv2.convertScaleAbs(rgb, alpha=alpha, beta=3.0)
                if response.width > self.live_preview_max_width:
                    scale = float(self.live_preview_max_width) / float(response.width)
                    rgb = cv2.resize(
                        rgb,
                        (self.live_preview_max_width, max(1, int(response.height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, jpeg = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.live_preview_quality],
                )
                if not ok:
                    return None
                with self._live_preview_lock:
                    index = self._live_preview_index
                    self._live_preview_index += 1
                final_path = self.live_preview_dir / f"preview_{index:08d}.jpg"
                temp_path = self.live_preview_dir / f"preview_{index:08d}.tmp"
                with temp_path.open("wb") as fp:
                    fp.write(jpeg.tobytes())
                os.replace(str(temp_path), str(final_path))
                return depth_guard
            except Exception as exc:
                self._live_preview_failures += 1
                logger.warning("Live preview capture failed (%d/3): %s", self._live_preview_failures, exc)
                return None

        def _setPoses(airsim_client: airsim.VehicleClient, pose: airsim.Pose) -> None:
            if airsim_client is None:
                raise Exception('error')
                return

            if self.live_preview_dir is None:
                airsim_client.simSetVehiclePose(
                    pose=pose,
                    ignore_collision=True,
                    vehicle_name='Drone_1',
                )
                return

            start_pose = airsim_client.simGetVehiclePose(vehicle_name='Drone_1')
            dx = float(pose.position.x_val - start_pose.position.x_val)
            dy = float(pose.position.y_val - start_pose.position.y_val)
            dz = float(pose.position.z_val - start_pose.position.z_val)
            distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            start_q = np.array([
                start_pose.orientation.x_val,
                start_pose.orientation.y_val,
                start_pose.orientation.z_val,
                start_pose.orientation.w_val,
            ], dtype=np.float64)
            target_q = np.array([
                pose.orientation.x_val,
                pose.orientation.y_val,
                pose.orientation.z_val,
                pose.orientation.w_val,
            ], dtype=np.float64)
            orientation_changed = 1.0 - abs(float(np.dot(start_q, target_q))) > 1e-6
            moving = distance > 1e-3
            _, _, start_yaw = airsim.to_eularian_angles(start_pose.orientation)
            horizontal_distance = float(np.sqrt(dx * dx + dy * dy))
            forward_motion = bool(
                horizontal_distance > 1e-3
                and (dx * np.cos(start_yaw) + dy * np.sin(start_yaw)) / horizontal_distance > 0.35
            )
            # Only the first placement is allowed to bypass sweep collisions; all navigation is solid.
            initial_reset = not self._live_pose_initialized
            if moving or orientation_changed:
                if distance <= self.live_preview_max_tween_distance:
                    distance_steps = int(np.ceil(max(distance, 0.0) / self.live_preview_meters_per_tween))
                    tween_steps = min(
                        self.live_preview_max_tween_steps,
                        max(self.live_preview_steps, distance_steps, 1),
                    )
                else:
                    tween_steps = 1
            else:
                tween_steps = 1
            with self._live_preview_lock:
                verify_pose = (
                    not self.live_fast_pose_verify
                    or initial_reset
                    or self._live_pose_verify_index % self.live_fast_pose_verify_interval == 0
                )
                self._live_pose_verify_index += 1
            safe_pose = copy.deepcopy(start_pose)
            try:
                collision_before = airsim_client.simGetCollisionInfo(vehicle_name='Drone_1')
                collision_before_time = int(getattr(collision_before, "time_stamp", 0) or 0)
            except Exception:
                collision_before_time = 0
            for tween_index in range(1, tween_steps + 1):
                tween_pose = _interpolate_pose(start_pose, pose, tween_index / tween_steps)
                airsim_client.simSetVehiclePose(
                    pose=tween_pose,
                    ignore_collision=initial_reset,
                    vehicle_name='Drone_1',
                )
                # AirSim normally applies a single non-colliding pose exactly.
                # Avoid an extra RPC on the fast path; collision events and a
                # periodic exact pose read still force the conservative path.
                actual_pose = (
                    airsim_client.simGetVehiclePose(vehicle_name='Drone_1')
                    if verify_pose
                    else copy.deepcopy(tween_pose)
                )
                if not initial_reset:
                    try:
                        collision_after = airsim_client.simGetCollisionInfo(vehicle_name='Drone_1')
                    except Exception:
                        collision_after = None
                    collision_after_time = int(getattr(collision_after, "time_stamp", 0) or 0)
                    new_collision = bool(
                        collision_after is not None
                        and getattr(collision_after, "has_collided", False)
                        and collision_after_time > collision_before_time
                    )
                    pose_blocked = False
                    if verify_pose:
                        actual_position = np.asarray(_pose_position(actual_pose), dtype=np.float64)
                        requested_position = np.asarray(_pose_position(tween_pose), dtype=np.float64)
                        pose_blocked = moving and float(np.linalg.norm(actual_position - requested_position)) > 0.35
                    if new_collision or pose_blocked:
                        if not verify_pose:
                            actual_pose = airsim_client.simGetVehiclePose(vehicle_name='Drone_1')
                        airsim_client.simSetVehiclePose(
                            pose=safe_pose,
                            ignore_collision=True,
                            vehicle_name='Drone_1',
                        )
                        _record_collision(
                            "airsim_collision" if new_collision else "pose_sweep_blocked",
                            collision_after,
                            tween_pose,
                            safe_pose,
                        )
                        actual_pose = copy.deepcopy(safe_pose)
                        _capture_preview(airsim_client)
                        break
                    collision_before_time = max(collision_before_time, collision_after_time)
                capture_frame = self.live_preview_capture_every_tween or (
                    not self.live_preview_capture_in_observation and tween_index == tween_steps
                )
                depth_guard = _capture_preview(
                    airsim_client,
                    include_depth=self.live_preview_depth_guard and forward_motion and not initial_reset,
                ) if capture_frame or self.live_preview_depth_guard else None
                visual_surface_blocked = bool(
                    depth_guard
                    and (
                        float(depth_guard["center_p10_m"]) < 1.25
                        or float(depth_guard["center_under_1m_ratio"]) >= 0.02
                    )
                )
                if visual_surface_blocked:
                    airsim_client.simSetVehiclePose(
                        pose=safe_pose,
                        ignore_collision=True,
                        vehicle_name='Drone_1',
                    )
                    _record_collision("visual_depth_guard", None, tween_pose, safe_pose)
                    actual_pose = copy.deepcopy(safe_pose)
                    _capture_preview(airsim_client)
                    break
                safe_pose = copy.deepcopy(actual_pose)

            _sync_pose(pose, actual_pose)
            self._live_pose_initialized = True

            return

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_setPoses, (self.airsim_clients[index_1][index_2], poses[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].get_result()
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('setPoses failed')
            return False

        return True

    def consumeCollisionEvent(self):
        with self._collision_lock:
            event = self.last_collision_event
            self.last_collision_event = None
        return event

    def restorePoses(self, poses: list) -> bool:
        """Restore a previously verified pose after a visual occupancy violation."""
        try:
            for index_1, clients in enumerate(self.airsim_clients):
                for index_2, airsim_client in enumerate(clients):
                    if airsim_client is None:
                        return False
                    pose = poses[index_1][index_2]
                    airsim_client.simSetVehiclePose(
                        pose=pose,
                        ignore_collision=True,
                        vehicle_name='Drone_1',
                    )
            return True
        except Exception as exc:
            logger.error("Failed to restore safe pose: %s", exc)
            return False

    def closeScenes(self):
        try:
            socket_clients = []
            for index, item in enumerate(self.machines_info):
                socket_clients.append(
                    msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=180)
                )

            for socket_client in socket_clients:
                if not self._confirmSocketConnection(socket_client):
                    logger.error('cannot establish socket')
                    raise Exception('cannot establish socket')

            self.socket_clients = socket_clients


            self._closeConnection()

            def _run_command(index, socket_client: msgpackrpc.Client):
                logger.info(f'START closing all scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}')
                result = socket_client.call('close_scenes', socket_client.address._host)
                logger.info(f'END closing all scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}')
                return

            threads = []
            for index, socket_client in enumerate(socket_clients):
                threads.append(
                    MyThread(_run_command, (index, socket_client))
                )
            for thread in threads:
                thread.setDaemon(True)
                thread.start()
            for thread in threads:
                thread.join()
            threads = []

            self._closeSocketConnection()
        except Exception as e:
            logger.error(e)


if __name__ == '__main__':

    machines_info_xxx = [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 1,
            'open_scenes': [1],
        },
    ]
    # machines_info_xxx = [
    #     {
    #         'MACHINE_IP': '127.0.0.1',
    #         'SOCKET_PORT': 30000,
    #         'MAX_SCENE_NUM': 8,
    #         'open_scenes': [1, 2, 3, 4, 5, 6, 7, None],
    #     },
    # ]

    tool = AirVLNSimulatorClientTool(machines_info=machines_info_xxx)
    tool.run_call()

    start_time = time.time()
    while True:
        time_1 = time.time()
        responses = tool.getImageResponses()
        time_2 = time.time()
        print(
            "total_time: {} \t time: {} \t fps: {}".format(
                (time_2-start_time),
                (time_2-time_1),
                1/(time_2-time_1),
            )
        )

        poses = []
        for index_1, item in enumerate(machines_info_xxx):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                pose=airsim.Pose(
                    position_val=airsim.Vector3r(random.randint(0, 1000), random.randint(0, 1000), random.randint(-200, 0)),
                    orientation_val=airsim.Quaternionr(0, 0, 0, 1),
                )
                poses[index_1].append(pose)

        tool.setPoses(poses)

