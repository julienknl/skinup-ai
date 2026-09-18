from fastapi import APIRouter, UploadFile, File, HTTPException
from service.recommendation_service import recommend_products
from PIL import Image
from io import BytesIO

router = APIRouter()

@router.post("/recommend_product")
async def recommend_product(file: UploadFile= File(...)):
    try:
        allowed_type = ["image/jpeg", "image/png"]
        if file.content_type not in allowed_type:
                raise ValueError("Only images with format JPG and PNG are allowed.")
        
        content = await file.read()
        
        image = Image.open(BytesIO(content)).convert("RGB")
        result = recommend_products(image=image)

        return { "result" : result}
    
    except ValueError as e:
        raise(HTTPException(status_code=400, detail=str(e)))

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
