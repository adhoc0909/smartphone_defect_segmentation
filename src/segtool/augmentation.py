import os
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import fire
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types
from PIL import Image
import io
import time
import random
import re
import cv2
import numpy as np

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class NanoAugmentor:
    def __init__(self, config: Dict):
        self.config = config
        pass

    def load_config(self, config_path: str) -> Dict:
        """Augentation settings load from json file."""
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return self.get_default_config()
        
    def save_config(self, config_path: str):
        """Save current configuration to a json file."""
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        # logger.info(f"Configuration saved to {config_path}") 









class CVAugmentor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self.get_default_config()

    def get_default_config(self) -> Dict:
        """Get default scratch augmentation configuration."""
        return {
            "scratch_augmentation": {
                "enabled": True,
                "num_scratches_range": [5, 12],  # 스크래치 개수 증가
                "alpha_range": [0.05, 0.12],  # 더 투명하게 (잔기스처럼)
                "screen_margin": {
                    "height_ratio": 0.25,
                    "width_ratio": 0.2
                },
                "scratch_types": {
                    "horizontal": {
                        "weight": 20,
                        "thickness_range": [1, 2],
                        "length_range": [80, 200]
                    },
                    "diagonal": {
                        "weight": 60,
                        "thickness_range": [1, 2],
                        "length_range": [50, 120]
                    },
                    "curved": {
                        "weight": 20,
                        "thickness": 1,
                        "control_point_variance": 50
                    }
                }
            },
            "basic_augmentation": {
                "enabled": False,  # 기본 증강 활성화
                "brightness_range": [0.8, 1.2],
                "contrast_range": [0.8, 1.2], 
                "rotation_range": [-5, 5],  # 도 단위
                "shadow_probability": 0.3
            },
            "labeling": {
                "mode": "preserve",  # "preserve" 또는 "new_class"
                "new_class_value": 113,  # 새 클래스 생성 시 사용할 픽셀값
                "target_classes": ["good"]  # 새 클래스 생성을 적용할 클래스들
            },
            "multi_stage": {
                "enabled": False,  # 다단계 증강 활성화
                "target_count": 400,  # 최종 스크래치 클래스 목표 수량
                "stage1_ratio": 19,  # good 클래스 기본 증강 배수 (자동 계산됨)
                "stage2_scratch_ratio": 1.0,  # 증강된 good에 스크래치 추가 비율
                "stage3_scratch_ratio": 1.0   # 원본 good에 스크래치 추가 비율
            }
        }

    def load_config(self, config_path: str) -> Dict:
        """Augentation settings load from json file."""
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return self.get_default_config()
        
    def save_config(self, config_path: str):
        """Save current configuration to a json file."""
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        logger.info(f"Configuration saved to {config_path}")

    def apply_scratch_augmentation(self, image: np.ndarray, return_mask: bool = False) -> np.ndarray:
        """
        Apply scratch augmentation to input image.
        
        Args:
            image: Input image as numpy array (BGR format)
            return_mask: Whether to return scratch mask along with image
            
        Returns:
            Augmented image with scratches, optionally with scratch mask
        """
        if not self.config["scratch_augmentation"]["enabled"]:
            if return_mask:
                return image, np.zeros(image.shape[:2], dtype=np.uint8)
            return image
            
        height, width = image.shape[:2]
        augmented_img = image.copy()
        scratch_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get configuration values
        scratch_config = self.config["scratch_augmentation"]
        margin_config = scratch_config["screen_margin"]
        types_config = scratch_config["scratch_types"]
        
        # Define screen area (phone screen region)
        screen_margin_h = int(height * margin_config["height_ratio"])
        screen_margin_w = int(width * margin_config["width_ratio"])
        screen_top = screen_margin_h
        screen_bottom = height - screen_margin_h
        screen_left = screen_margin_w
        screen_right = width - screen_margin_w
        
        # Number of scratches
        num_scratches_range = scratch_config["num_scratches_range"]
        num_scratches = random.randint(num_scratches_range[0], num_scratches_range[1])
        
        # Generate scratches
        for _ in range(num_scratches):
            # Choose scratch type with weighted probability
            scratch_type = random.choices(
                ['horizontal', 'diagonal', 'curved'],
                weights=[
                    types_config['horizontal']['weight'],
                    types_config['diagonal']['weight'],
                    types_config['curved']['weight']
                ]
            )[0]
            
            # Set transparency
            alpha_range = scratch_config["alpha_range"]
            alpha = random.uniform(alpha_range[0], alpha_range[1])
            
            if scratch_type == 'horizontal':
                augmented_img, scratch_mask = self._add_horizontal_scratch(
                    augmented_img, screen_top, screen_bottom, screen_left, screen_right,
                    types_config['horizontal'], alpha, scratch_mask
                )
            elif scratch_type == 'diagonal':
                augmented_img, scratch_mask = self._add_diagonal_scratch(
                    augmented_img, screen_top, screen_bottom, screen_left, screen_right,
                    types_config['diagonal'], alpha, scratch_mask
                )
            elif scratch_type == 'curved':
                augmented_img, scratch_mask = self._add_curved_scratch(
                    augmented_img, screen_top, screen_bottom, screen_left, screen_right,
                    types_config['curved'], alpha, scratch_mask
                )
        
        if return_mask:
            return augmented_img, scratch_mask
        return augmented_img

    def _add_horizontal_scratch(self, image: np.ndarray, screen_top: int, screen_bottom: int, 
                              screen_left: int, screen_right: int, config: Dict, alpha: float, mask: np.ndarray) -> tuple:
        """Add horizontal scratch to image and mask."""
        height, width = image.shape[:2]
        
        y = random.randint(screen_top, screen_bottom - 20)
        x_start = random.randint(screen_left, screen_right - 150)
        scratch_length = random.randint(config["length_range"][0], config["length_range"][1])
        x_end = min(x_start + scratch_length, screen_right)
        thickness = random.randint(config["thickness_range"][0], config["thickness_range"][1])
        
        for i in range(thickness):
            if y + i < height:
                # Add scratch to image
                original_color = image[y+i:y+i+1, x_start:x_end]
                scratch_color = np.full_like(original_color, [255, 255, 255])
                blended = (1 - alpha) * original_color + alpha * scratch_color
                image[y+i:y+i+1, x_start:x_end] = blended.astype(np.uint8)
                
                # Add scratch to mask (white = scratch area)
                mask[y+i:y+i+1, x_start:x_end] = 255
        
        return image, mask

    def _add_diagonal_scratch(self, image: np.ndarray, screen_top: int, screen_bottom: int,
                            screen_left: int, screen_right: int, config: Dict, alpha: float, mask: np.ndarray) -> tuple:
        """Add diagonal scratch to image and mask."""
        height, width = image.shape[:2]
        
        x_start = random.randint(screen_left, screen_right - 100)
        y_start = random.randint(screen_top, screen_bottom - 100)
        length = random.randint(config["length_range"][0], config["length_range"][1])
        thickness = random.randint(config["thickness_range"][0], config["thickness_range"][1])
        
        # Direction
        direction_x = random.choice([-1, 1])
        direction_y = random.choice([-1, 1])
        
        for i in range(length):
            x = x_start + i * direction_x
            y = y_start + i * direction_y
            
            if screen_left <= x < screen_right and screen_top <= y < screen_bottom:
                for tx in range(-thickness//2, thickness//2 + 1):
                    for ty in range(-thickness//2, thickness//2 + 1):
                        if 0 <= x+tx < width and 0 <= y+ty < height:
                            # Add scratch to image
                            original = image[y+ty, x+tx]
                            scratch = np.array([255, 255, 255])
                            blended = (1 - alpha) * original + alpha * scratch
                            image[y+ty, x+tx] = blended.astype(np.uint8)
                            
                            # Add scratch to mask
                            mask[y+ty, x+tx] = 255
        
        return image, mask

    def _add_curved_scratch(self, image: np.ndarray, screen_top: int, screen_bottom: int,
                          screen_left: int, screen_right: int, config: Dict, alpha: float, mask: np.ndarray) -> tuple:
        """Add curved scratch to image and mask using Bezier curve."""
        height, width = image.shape[:2]
        
        x_start = random.randint(screen_left, screen_right - 100)
        y_start = random.randint(screen_top, screen_bottom - 100)
        
        # Control points
        variance = config["control_point_variance"]
        x_mid = x_start + random.randint(-variance, variance)
        y_mid = y_start + random.randint(-variance, variance)
        x_end = x_start + random.randint(40, 100)
        y_end = y_start + random.randint(-40, 40)
        
        # Constrain to screen area
        x_mid = max(screen_left, min(x_mid, screen_right))
        y_mid = max(screen_top, min(y_mid, screen_bottom))
        x_end = max(screen_left, min(x_end, screen_right))
        y_end = max(screen_top, min(y_end, screen_bottom))
        
        # Generate curve points
        points = []
        for t in np.linspace(0, 1, 60):
            x = int((1-t)**2 * x_start + 2*(1-t)*t * x_mid + t**2 * x_end)
            y = int((1-t)**2 * y_start + 2*(1-t)*t * y_mid + t**2 * y_end)
            if screen_left <= x < screen_right and screen_top <= y < screen_bottom:
                points.append((x, y))
        
        # Draw curve
        thickness = config["thickness"]
        for x, y in points:
            if 0 <= x < width and 0 <= y < height:
                # Add scratch to image
                original = image[y, x]
                scratch = np.array([255, 255, 255])
                blended = (1 - alpha) * original + alpha * scratch
                image[y, x] = blended.astype(np.uint8)
                
                # Add scratch to mask
                mask[y, x] = 255
        
        return image, mask

    def apply_basic_augmentation(self, image: np.ndarray) -> np.ndarray:
        """기본 증강 기법 적용 (밝기, 대비, 회전, 그림자)"""
        if not self.config.get("basic_augmentation", {}).get("enabled", False):
            return image
            
        aug_config = self.config["basic_augmentation"]
        augmented = image.copy().astype(np.float32)
        
        # 밝기 조정
        brightness = random.uniform(*aug_config["brightness_range"])
        augmented = augmented * brightness
        
        # 대비 조정
        contrast = random.uniform(*aug_config["contrast_range"])
        mean = augmented.mean()
        augmented = (augmented - mean) * contrast + mean
        
        # 회전
        angle = random.uniform(*aug_config["rotation_range"])
        if abs(angle) > 0.1:
            h, w = augmented.shape[:2]
            center = (w//2, h//2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, matrix, (w, h))
        
        # 그림자 효과
        if random.random() < aug_config["shadow_probability"]:
            augmented = self._add_shadow(augmented)
            
        # 클리핑
        augmented = np.clip(augmented, 0, 255).astype(np.uint8)
        return augmented
    
    def _add_shadow(self, image: np.ndarray) -> np.ndarray:
        """간단한 그림자 효과 추가"""
        h, w = image.shape[:2]
        
        # 그림자 영역 생성 (타원형)
        shadow_mask = np.zeros((h, w), dtype=np.float32)
        center_x = random.randint(w//4, 3*w//4)
        center_y = random.randint(h//4, 3*h//4)
        radius_x = random.randint(w//6, w//3)
        radius_y = random.randint(h//6, h//3)
        
        cv2.ellipse(shadow_mask, (center_x, center_y), (radius_x, radius_y), 
                   0, 0, 360, 1.0, -1)
        
        # 그림자 강도
        shadow_intensity = random.uniform(0.3, 0.7)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (51, 51), 0)
        
        # 그림자 적용
        for c in range(3):
            image[:,:,c] = image[:,:,c] * (1 - shadow_mask * shadow_intensity)
            
        return image

    def augment_image(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Apply augmentation to a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save augmented image (optional)
            
        Returns:
            Augmented image as numpy array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Apply scratch augmentation
        augmented = self.apply_scratch_augmentation(image)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, augmented)
            logger.info(f"Augmented image saved to {output_path}")
        
        return augmented

    def augment_batch(self, image_paths: List[str], output_dir: str) -> List[str]:
        """
        Apply augmentation to a batch of images.
        
        Args:
            image_paths: List of input image paths
            output_dir: Directory to save augmented images
            
        Returns:
            List of output image paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for image_path in image_paths:
            # Create output path
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_augmented{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Apply augmentation
            try:
                self.augment_image(image_path, output_path)
                output_paths.append(output_path)
            except Exception as e:
                logger.error(f"Failed to augment {image_path}: {e}")
        
        logger.info(f"Batch augmentation completed. {len(output_paths)}/{len(image_paths)} images processed.")
        return output_paths

    def augment_dataset_with_ratio(self, input_dir: str, output_dir: str, 
                                  augment_ratios: Dict[str, float] = None,
                                  augment_counts: Dict[str, int] = None,
                                  exclude_classes: List[str] = None):
        """
        클래스별 비율 또는 개수에 따라 데이터셋 증강
        
        Args:
            input_dir: 입력 데이터셋 디렉토리
            output_dir: 출력 디렉토리 
            augment_ratios: 클래스별 증강 비율 딕셔너리 (예: {"good": 0.5, "oil": 0.3})
                           0.5 = 50% 이미지만 증강, None이면 전체 증강
            augment_counts: 클래스별 증강 개수 딕셔너리 (예: {"good": 50, "oil": 100})
                           ratios보다 우선순위 높음
            exclude_classes: 제외할 클래스 리스트
        """
        exclude_classes = exclude_classes or []
        augment_ratios = augment_ratios or {}
        augment_counts = augment_counts or {}
        
        for class_name in os.listdir(input_dir):
            class_path = os.path.join(input_dir, class_name)
            
            # 디렉토리가 아니거나 제외 클래스면 스킵
            if not os.path.isdir(class_path) or class_name in exclude_classes:
                continue
                
            # aug 폴더 및 aug_mask 폴더 생성
            aug_dir = os.path.join(output_dir, "aug")
            aug_mask_dir = os.path.join(output_dir, "aug_mask")
            os.makedirs(aug_dir, exist_ok=True)
            os.makedirs(aug_mask_dir, exist_ok=True)
            
            # 클래스의 모든 이미지 파일 가져오기
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # 증강할 개수 결정 (count가 ratio보다 우선순위 높음)
            if class_name in augment_counts:
                num_to_augment = min(augment_counts[class_name], len(image_files))
                mode = "count"
                value = augment_counts[class_name]
            elif class_name in augment_ratios:
                ratio = augment_ratios[class_name]
                num_to_augment = int(len(image_files) * ratio)
                mode = "ratio"
                value = ratio
            else:
                # counts와 ratios 모두에 없으면 증강 안함
                logger.info(f"Class {class_name}: Skipping (not specified in counts or ratios)")
                continue
            
            if num_to_augment == 0:
                if mode == "count":
                    logger.info(f"Class {class_name}: Skipping (count: {value})")
                else:
                    logger.info(f"Class {class_name}: Skipping (ratio: {value*100:.1f}%)")
                continue
            
            # 랜덤하게 선택
            selected_files = random.sample(image_files, num_to_augment)
            
            if mode == "count":
                logger.info(f"Class {class_name}: Augmenting {num_to_augment}/{len(image_files)} images (count: {value})")
            else:
                logger.info(f"Class {class_name}: Augmenting {num_to_augment}/{len(image_files)} images ({value*100:.1f}%)")
            
            for img_file in selected_files:
                input_path = os.path.join(class_path, img_file)
                
                # 파일명 생성: aug_{class_name}_{original_filename}
                name, ext = os.path.splitext(img_file)
                augmented_filename = f"aug_{class_name}_{img_file}"
                output_path = os.path.join(aug_dir, augmented_filename)
                mask_output_path = os.path.join(aug_mask_dir, augmented_filename.replace(ext, '.png'))
                
                try:
                    # 기존 마스크 경로 찾기
                    original_mask_path = self._find_original_mask(class_name, img_file, input_dir)
                    
                    # 이미지와 마스크 동시 증강
                    self._augment_image_with_mask(input_path, output_path, mask_output_path, 
                                                original_mask_path, class_name)
                except Exception as e:
                    logger.error(f"Failed to process {input_path}: {e}")
            
            logger.info(f"Completed augmentation for class {class_name}")

    def multi_stage_augmentation(self, input_dir: str, output_dir: str):
        """
        다단계 증강: 목표 수량(400개)에 맞춰 새로운 스크래치 클래스 생성
        """
        multi_config = self.config.get("multi_stage", {})
        if not multi_config.get("enabled", False):
            logger.info("Multi-stage augmentation disabled")
            return
            
        target_count = multi_config.get("target_count", 400)
        basic_aug_ratio = multi_config.get("basic_aug_ratio", 0.2)
        
        good_dir = Path(input_dir) / "good"
        aug_dir = Path(output_dir) / "aug"
        aug_mask_dir = Path(output_dir) / "aug_mask"
        
        os.makedirs(aug_dir, exist_ok=True)
        os.makedirs(aug_mask_dir, exist_ok=True)
        
        good_images = [f for f in os.listdir(good_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        good_count = len(good_images)
        
        # 단순화: 총 400개만 생성 (기본 증강 + 스크래치)
        basic_aug_count = int(target_count * basic_aug_ratio)  # 기본 증강 이미지 수
        scratch_total_count = target_count - basic_aug_count  # 스크래치 이미지 수
        
        # 각 단계별 생성 개수 계산
        stage1_ratio = max(1, basic_aug_count // good_count + (1 if basic_aug_count % good_count > 0 else 0))
        stage3_ratio = max(1, scratch_total_count // good_count + (1 if scratch_total_count % good_count > 0 else 0))
        
        logger.info(f"Target: {target_count} scratch images")
        logger.info(f"Good images: {good_count}")
        logger.info(f"Basic aug ratio: {basic_aug_ratio} ({basic_aug_count} images)")
        logger.info(f"Stage1: {basic_aug_count} basic augmented images (ratio: {stage1_ratio})")
        logger.info(f"Stage2: SKIP - 단순화된 로직")
        logger.info(f"Stage3: {scratch_total_count} scratch images (ratio: {stage3_ratio})")
        
        # 1단계: Good 클래스 기본 증강
        logger.info(f"Stage 1: Creating basic augmented images")
        self.config["basic_augmentation"]["enabled"] = True
        stage1_images = []
        
        aug_types = ["bright", "contrast", "rotate", "shadow", "mixed"]
        images_per_type = stage1_ratio // len(aug_types) + 1
        
        stage1_count = 0
        for img_file in good_images:
            for i in range(stage1_ratio):
                if stage1_count >= basic_aug_count:
                    break
                    
                input_path = good_dir / img_file
                name, ext = os.path.splitext(img_file)
                aug_type = aug_types[i % len(aug_types)]
                output_filename = f"aug_good_{aug_type}_{name}_{i+1:04d}{ext}"
                output_path = aug_dir / output_filename
                
                image = cv2.imread(str(input_path))
                augmented = self.apply_basic_augmentation(image)
                cv2.imwrite(str(output_path), augmented)
                stage1_images.append(output_filename)
                stage1_count += 1
                
            if stage1_count >= basic_aug_count:
                break
                
        logger.info(f"Stage 1 completed: {len(stage1_images)} basic augmented images")
        
        # 2단계: 증강된 Good → 스크래치 추가 (덮어쓰기로 총 개수 유지)
        logger.info(f"Stage 2: Adding scratches to augmented images (overwrite)")
        self.config["labeling"]["mode"] = "new_class"
        
        # 모든 기본 증강 이미지를 스크래치로 변환
        for idx, img_file in enumerate(stage1_images):
            input_path = aug_dir / img_file
            # 같은 파일명으로 덮어쓰기
            output_path = input_path
            
            # 마스크 파일명 생성
            name, ext = os.path.splitext(img_file)
            mask_output_path = aug_mask_dir / f"{name}.png"
            
            self._augment_image_with_mask(str(input_path), str(output_path),
                                        str(mask_output_path), None, "good")
                                        
        logger.info(f"Stage 2 completed: {len(stage1_images)} images converted to scratch")
        
        # 3단계: 원본 Good → 스크래치 추가  
        logger.info(f"Stage 3: Adding scratches to original images")
        
        stage3_count = 0
        for img_file in good_images:
            for i in range(stage3_ratio):
                if stage3_count >= scratch_total_count:
                    break
                    
                input_path = good_dir / img_file
                name, ext = os.path.splitext(img_file)
                output_filename = f"aug_good_scratch_{name}_{i+1:04d}{ext}"
                output_path = aug_dir / output_filename
                mask_output_path = aug_mask_dir / f"aug_good_scratch_{name}_{i+1:04d}.png"
                
                self._augment_image_with_mask(str(input_path), str(output_path),
                                            str(mask_output_path), None, "good")
                stage3_count += 1
            
            if stage3_count >= scratch_total_count:
                break
                                        
        total_images = len(stage1_images) + stage3_count
        total_scratch_images = len(stage1_images) + stage3_count  # Stage2에서 모든 기본 증강이 스크래치로 변환됨
        logger.info(f"Stage 3 completed: {stage3_count} scratch images from originals")
        logger.info(f"Multi-stage augmentation completed: {total_images} total images")
        logger.info(f"Basic augmented (converted to scratch): {len(stage1_images)}")
        logger.info(f"Additional scratch images: {stage3_count}")
        logger.info(f"Total scratch images: {total_scratch_images}")
        logger.info(f"All files saved to 'aug/' folder with proper naming")

    def _find_original_mask(self, class_name: str, img_file: str, input_dir: str) -> Optional[str]:
        """
        Find the original mask file for a given image.
        
        Args:
            class_name: Class name (good, oil, stain, scratch)
            img_file: Image filename
            input_dir: Input directory path
            
        Returns:
            Path to mask file or None if not found
        """
        # good 클래스는 마스크가 없음
        if class_name == "good":
            return None
            
        # 파일명에서 확장자 제거
        base_name = os.path.splitext(img_file)[0]
        mask_name = base_name + '.png'
        
        # 클래스별 마스크 경로
        if class_name in ["stain", "scratch"]:
            # ground_truth_1에서 찾기
            mask_path = os.path.join(input_dir, "ground_truth_1", mask_name)
        elif class_name == "oil":
            # ground_truth_2에서 찾기  
            mask_path = os.path.join(input_dir, "ground_truth_2", mask_name)
        else:
            return None
            
        return mask_path if os.path.exists(mask_path) else None

    def _augment_image_with_mask(self, input_path: str, output_path: str, mask_output_path: str, 
                               original_mask_path: Optional[str], class_name: str):
        """
        Apply augmentation to image and generate corresponding mask.
        
        Args:
            input_path: Input image path
            output_path: Output image path
            mask_output_path: Output mask path
            original_mask_path: Original mask path (can be None)
            class_name: Class name
        """
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image from {input_path}")
        
        # Apply scratch augmentation with mask generation
        augmented_image, scratch_mask = self.apply_scratch_augmentation(image, return_mask=True)
        
        # Load original mask if exists
        if original_mask_path and os.path.exists(original_mask_path):
            original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
            if original_mask is not None:
                # Resize original mask to match image size if needed
                if original_mask.shape != scratch_mask.shape:
                    original_mask = cv2.resize(original_mask, 
                                             (scratch_mask.shape[1], scratch_mask.shape[0]))
                # Combine original mask and scratch mask
                combined_mask = np.maximum(original_mask, scratch_mask)
            else:
                combined_mask = scratch_mask
        else:
            # No original mask, use only scratch mask
            combined_mask = scratch_mask
        
        # 후처리: 255 값을 올바른 클래스 값으로 변경
        combined_mask = self._post_process_mask(combined_mask, class_name, original_mask_path)
        
        # Save augmented image and mask
        cv2.imwrite(output_path, augmented_image)
        cv2.imwrite(mask_output_path, combined_mask)
        
        logger.info(f"Saved augmented image: {output_path}")
        logger.info(f"Saved augmented mask: {mask_output_path}")

    def _post_process_mask(self, mask: np.ndarray, class_name: str, original_mask_path: Optional[str] = None) -> np.ndarray:
        """
        Post-process mask to fix incorrect pixel values.
        동적으로 기존 라벨 유지 또는 새 클래스 생성 가능
        """
        processed_mask = mask.copy()
        labeling_config = self.config.get("labeling", {})
        mode = labeling_config.get("mode", "preserve")
        target_classes = labeling_config.get("target_classes", [])
        new_class_value = labeling_config.get("new_class_value", 113)
        
        # 새 클래스 생성 모드 & 대상 클래스인 경우
        if mode == "new_class" and class_name in target_classes:
            processed_mask[processed_mask == 255] = new_class_value
            # 보간값들도 새 클래스값으로 처리
            intermediate_mask = (processed_mask > 0) & (processed_mask != new_class_value)
            processed_mask[intermediate_mask] = new_class_value
            
            logger.debug(f"New class mode: {class_name} -> new class value {new_class_value}")
        
        # 기존 라벨 유지 모드 (기본)
        elif class_name == 'good':
            processed_mask[processed_mask == 255] = 0
        else:
            class_value = self._extract_class_value(original_mask_path, class_name)
            if class_value is not None:
                processed_mask[processed_mask == 255] = class_value
                intermediate_mask = (processed_mask > 0) & (processed_mask != class_value)
                processed_mask[intermediate_mask] = class_value
        
        logger.debug(f"Mask processing for {class_name} ({mode} mode): "
                    f"{np.unique(mask)} -> {np.unique(processed_mask)}")
        
        return processed_mask

    def _extract_class_value(self, original_mask_path: Optional[str], class_name: str) -> Optional[int]:
        """
        Extract class pixel value from original mask.
        
        Args:
            original_mask_path: Path to original mask file
            class_name: Class name for logging
            
        Returns:
            Class pixel value or None if not found
        """
        if not original_mask_path or not os.path.exists(original_mask_path):
            logger.warning(f"Original mask not found for {class_name}")
            return None
            
        try:
            original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
            if original_mask is None:
                logger.warning(f"Could not load original mask: {original_mask_path}")
                return None
                
            # 고유한 픽셀값 추출 (0 제외)
            unique_values = np.unique(original_mask)
            class_values = unique_values[unique_values > 0]
            
            if len(class_values) > 0:
                # 일반적으로 가장 큰 값이 클래스 값
                class_value = class_values[-1]
                logger.debug(f"Extracted class value {class_value} for {class_name} from {original_mask_path}")
                return int(class_value)
            else:
                logger.warning(f"No class values found in {original_mask_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting class value from {original_mask_path}: {e}")
            return None


def cli_augment_dataset(input_dir: str, output_dir: str = None, 
                       exclude_classes: str = "ground_truth_1,ground_truth_2",
                       ratios: str = None,
                       counts: str = None,
                       config_path: str = None,
                       labeling_mode: str = "preserve",
                       new_class_value: int = 113,
                       target_classes: str = "good",
                       multi_stage: bool = False,
                       target_count: int = 400,
                       basic_aug_ratio: float = 0.2):
    """
    CLI interface for dataset augmentation
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output directory (defaults to same as input_dir)
        exclude_classes: Comma-separated list of classes to exclude
        ratios: Comma-separated class:ratio pairs (e.g., "good:0.3,oil:0.5,stain:0.8")
        counts: Comma-separated class:count pairs (e.g., "good:50,oil:100") - 우선순위 높음
        config_path: Path to configuration file
        labeling_mode: "preserve" (기존 라벨 유지) or "new_class" (새 클래스 생성)
        new_class_value: 새 클래스 생성 시 사용할 픽셀값 (default: 113)
        target_classes: 새 클래스 생성을 적용할 클래스들 (default: "good")
        multi_stage: 다단계 증강 활성화 (default: False)
        target_count: 새 스크래치 클래스 목표 수량 (default: 400)
        basic_aug_ratio: 기본 증강 이미지 비율 0.0~1.0 (default: 0.2 = 20%)
    
    Examples:
        # 기존 방식 (라벨 유지) - ratio로
        python -m fire augmentation.py cli_augment_dataset --input_dir="/path/to/data" --ratios="good:0.3"
        
        # count로 정확한 개수 지정
        python -m fire augmentation.py cli_augment_dataset --input_dir="/path/to/data" --counts="good:50,oil:100"
        
        # 다단계 증강 - 기본 증강 10%로 낮추기
        python -m fire augmentation.py cli_augment_dataset \
            --input_dir="/path/to/data" \
            --multi_stage=True \
            --target_count=400 \
            --basic_aug_ratio=0.1
    """
    
    # Parse parameters
    exclude_list = [cls.strip() for cls in exclude_classes.split(',')] if exclude_classes else []
    target_list = [cls.strip() for cls in target_classes.split(',')] if target_classes else ["good"]
    
    ratio_dict = {}
    if ratios:
        for ratio_pair in ratios.split(','):
            if ':' in ratio_pair:
                class_name, ratio_val = ratio_pair.split(':', 1)
                try:
                    ratio_dict[class_name.strip()] = float(ratio_val.strip())
                except ValueError:
                    logger.warning(f"Invalid ratio format: {ratio_pair}")
    
    count_dict = {}
    if counts:
        for count_pair in counts.split(','):
            if ':' in count_pair:
                class_name, count_val = count_pair.split(':', 1)
                try:
                    count_dict[class_name.strip()] = int(count_val.strip())
                except ValueError:
                    logger.warning(f"Invalid count format: {count_pair}")
    
    output_dir = output_dir or input_dir
    
    # Initialize augmentor with dynamic config
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = CVAugmentor().get_default_config()
    
    # 동적으로 설정 업데이트
    config["labeling"] = {
        "mode": labeling_mode,
        "new_class_value": new_class_value,
        "target_classes": target_list
    }
    config["multi_stage"] = {
        "enabled": multi_stage,
        "target_count": target_count,
        "basic_aug_ratio": basic_aug_ratio,
        "stage2_scratch_ratio": 1.0,
        "stage3_scratch_ratio": 1.0
    }
    
    augmentor = CVAugmentor(config)
    
    # Log configuration
    logger.info(f"Input: {input_dir} | Output: {output_dir}")
    logger.info(f"Multi-stage: {multi_stage} | Target count: {target_count}")
    logger.info(f"Labeling mode: {labeling_mode} | Target classes: {target_list}")
    
    # Run augmentation
    if multi_stage:
        augmentor.multi_stage_augmentation(input_dir=input_dir, output_dir=output_dir)
    else:
        logger.info(f"Ratios: {ratio_dict} | Counts: {count_dict} | Exclude: {exclude_list}")
        augmentor.augment_dataset_with_ratio(
            input_dir=input_dir,
            output_dir=output_dir,
            augment_ratios=ratio_dict,
            augment_counts=count_dict,
            exclude_classes=exclude_list
        )
    
    logger.info("Dataset augmentation completed!")


if __name__ == "__main__":
    fire.Fire()