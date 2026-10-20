cmake_minimum_required(VERSION 3.0.2)
project(navi_types)

# 找到catkin并设置依赖项
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  actionlib
  actionlib_msgs
  trajectory_msgs  
  message_generation
)


# 定义消息和服务文件
add_message_files(
  FILES
  ServoToolStatus.msg
  Robot_StateMsg.msg
  Robot_WorkStatus.msg
  nlu_result.msg
  entity.msg
    # 你定义的消息文件
)

add_message_files(
  FILES
  DetItem.msg
  Grasp6d.msg
  ObjPose.msg
  Strawberry.msg
)

add_service_files(
  FILES
  Hand_Gesture.srv
  Hand_Joint.srv
  Hand_Task.srv
  Uplimb_MoveJ.srv
  Uplimb_MoveL.srv
  Uplimb_TeachMode.srv
  Robot_SetState.srv
  Robot_BasicInfo.srv
  Robot_CameraInfo.srv
  Robot_HeadIMUInfo.srv
  Robot_WifiInfo.srv
  Robot_ConnectWifi.srv
  JointsSeqTracking.srv
  PointsSeqTracking.srv
  PointsSeqTrackingWithJoints.srv
  
)

add_service_files(
  FILES
  Audio_TTS.srv
  Audio_Listen.srv
  Audio_MediaPlay.srv
  Audio_GetVolume.srv
  Audio_SetDevice.srv
  Audio_SetVolume.srv
  Audio_GetDeviceList.srv
  Audio_VideoPlay.srv
  LLM_Chat.srv
  LLM_NLU.srv
  LLM_Session.srv
  )

add_service_files(
  FILES
  SetServoRef.srv
  RunServo.srv
  DualArmCollaborationTracking.srv
  Manip_GetScenePose.srv
  Manip_GetTrajectory.srv
  GraspConstr.srv
  GraspMotionPlan.srv
  GraspPlan.srv
  InstSeg.srv
  PlaceMotionPlan.srv
  PoseEst.srv
  DepthRefine.srv
  RunCmd.srv
  SceneUpdate.srv
  RobotHandJointSrv.srv
  LoadImageBase64.srv
  MotionPlan.srv
  MotionPlanMulti.srv
  TactileHold.srv
  TactileRelease.srv
  TrajExec.srv
  GetStrawberry.srv
  EpropnpEst.srv
  Grasp.srv
)


# Generate actions in the 'action' folder
add_action_files(
   FILES
   RunServoTool.action
   InstSeg.action
   Pick.action
   Place.action
   Search_object.action
   LoosenHand.action
   Track.action
)

# 生成消息和服务
generate_messages(
  DEPENDENCIES
  std_msgs  # 如果你的消息依赖于 std_msgs 等其他消息
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  actionlib_msgs
)

# 声明catkin包
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs message_runtime sensor_msgs actionlib actionlib_msgs trajectory_msgs # 必须声明生成消息所需的依赖
)

# 包含 ROS 头文件路径
include_directories(
  ${catkin_INCLUDE_DIRS}  # 关键！确保包含此路径
)

# 安装设置
install(DIRECTORY msg srv
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

