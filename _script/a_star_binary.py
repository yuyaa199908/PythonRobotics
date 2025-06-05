"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""
import cv2
import numpy as np

import math

import matplotlib.pyplot as plt

show_animation = True




class AStarPlanner:

    def __init__(self,resolution, rr, binary_image, wx, wy, map_grid_size):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(binary_image, wx, wy, map_grid_size)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, binary_image, wx, wy, map_grid_size):

        # 画像サイズを取得
        image_height, image_width = binary_image.shape

        # ワールド座標系での障害物マップの範囲を計算
        self.min_x = wx
        self.min_y = wy
        self.max_x = wx + image_width * map_grid_size
        self.max_y = wy + image_height * map_grid_size

        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        # グリッドサイズを計算
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # 障害物マップを初期化
        self.obstacle_map = [[False for _ in range(self.y_width)]
                            for _ in range(self.x_width)]

        # 障害物マップを生成
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)  # グリッドのワールド座標X
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)  # グリッドのワールド座標Y

                # ワールド座標を画像のピクセル座標に変換
                pixel_x = int((x - wx) / map_grid_size)
                pixel_y = int((y - wy) / map_grid_size)

                # 範囲外チェック
                if pixel_x < 0 or pixel_y < 0 or pixel_x >= image_width or pixel_y >= image_height:
                    continue

                # 障害物判定 (黒いピクセルが障害物)
                if binary_image[pixel_y, pixel_x] == 0:  # 黒いピクセルは障害物
                    self.obstacle_map[ix][iy] = True
        # print(self.obstacle_map)

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

def main():
    # start and goal position
    sx = 60.0  # [m]
    sy = 0.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 0.1  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    # ox, oy = [], []

    # PNG画像を読み込む
    image = cv2.imread("Untitled.png", cv2.IMREAD_GRAYSCALE)  # グレースケールで読み込み
    # 閾値を設定
    threshold_value = 128 
    # 二値化処理
    _, binary_image = cv2.threshold(image, threshold_value, 1, cv2.THRESH_BINARY)

    # world position of the image origin
    wx = -2.0   # [m]
    wy = -1.0   # [m]
    map_grid_size = 0.1 # [m]

    if show_animation:  # pragma: no cover
        # 障害物ピクセルの座標を取得
        obstacle_coords = np.argwhere(binary_image == 0)  # 黒いピクセル (障害物)
        world_coords_x = wx + obstacle_coords[:, 1] * map_grid_size  # x座標
        world_coords_y = wy + obstacle_coords[:, 0] * map_grid_size  # y座標
        plt.plot(world_coords_x, world_coords_y, ".k")
        
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        
        plt.grid(True)
        plt.axis("equal")
    
    a_star = AStarPlanner(grid_size, robot_radius, binary_image, wx, wy, map_grid_size)
    rx, ry = a_star.planning(sx, sy, gx, gy)


    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()
