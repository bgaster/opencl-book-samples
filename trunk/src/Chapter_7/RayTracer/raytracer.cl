//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// raytracer.cl
//
//    This is a (very) simple raytracer that is intended to demonstrate 
//    using OpenCL buffers.

#include "raytracer.hpp"

#pragma OPENCL EXTENSION cl_amd_printf : enable

#define BACKGROUND_COLOUR (float3)(0.0f, 0.0f, 0.0f)
#define DEFAULT_COLOUR (float3)(0.0f, 0.0f, 0.0f)

#define EPSILON 0.01f

uchar toUChar(float x) 
{
	return (uchar)(pow(clamp(x, 0.f, 1.f), 1.f / 2.2f) * 255.f + .5f);
}

float3 surfaceDiffuse(cl_float3 position, Surface surface)
{
	switch(surface.type_) 
	{
		case SURFACE_MATT_SHINY:
			return (float3)(1.0f,1.0f,1.0f);
			break;
		case SURFACE_SHINY:
			return (float3)(1.0f,1.0f,1.0f);
			break;
		case SURFACE_CHECKERBOARD:
		{
			if ((convert_int(floor(position.z)) + convert_int(floor(position.x))) % 2 != 0)
			{
				return (float3)(1.f, 1.f, 1.f);
			}
			else
			{
				return (float3)(0.02f, 0.0f, 0.14f);
			}
			break;
		}
	}
}

float3 surfaceSpecular(cl_float3 position, Surface surface)
{
	switch(surface.type_) 
	{
		case SURFACE_MATT_SHINY:
			return (float3)(0.25f, 0.25f, 0.25f);
			break;
		case SURFACE_SHINY:
			return (float3)(0.5f,0.5f,0.5f);
			break;
		case SURFACE_CHECKERBOARD:
			return (float3)(1.f,1.f,1.f);
			break;
	}
}

float surfaceReflect(cl_float3 position, Surface surface)
{
	switch(surface.type_) 
	{
		case SURFACE_MATT_SHINY:
			return 0.7f;
			break;
		case SURFACE_SHINY:
			return 0.7f;
			break;
		case SURFACE_CHECKERBOARD:
			if ((convert_int(floor(position.z)) + convert_int(floor(position.x))) % 2 != 0)
			{
				return 0.1f;
			}
			else
			{
				return 0.5f;
			}
			break;
	}
}

InterSection intersect(Object object, Ray ray)
{
	InterSection isection;

	switch(object.type_) 
	{
		case OBJECT_PLANE:
		{
			float dist;
			cl_float3 norm = object.centerOrNorm_;
			float offset   = object.radiusOrOffset_;

			float denom = dot(norm, ray.direction_);
			if (denom > 0)
			{
				dist = 0.0f;
			}
			else
			{
				dist = (dot(norm, ray.start_) + offset) / (-denom);
			}

			isection.object_ = object;
			isection.ray_    = ray;
			isection.dist_    = dist;
			break;
		}
		case OBJECT_SPHERE:
		{
			cl_float3 center = object.centerOrNorm_;
			float radius     = object.radiusOrOffset_;

			float3 eo      = center - ray.start_;

			float  v       = dot(eo, ray.direction_);

			float dist;

			if ( v < 0.0f)
			{
				dist = 0;
			}
			else
			{
				dist = (radius * radius) - (dot(eo,eo) - (v * v));
				if (dist < 0.0f)
				{
					dist = 0.0f;
				}
				else
				{
					dist = v - (sqrt(dist));
				}
			}

#if 0
			float dist = v * v - dot(eo,eo) + radius * radius;

			if (dist  < 0.f)
			{
				dist = 0.0f;
			}
			else
			{
				dist = sqrt(dist);

				float t = v - dist;
				if (t > EPSILON)
				{
					dist = t;
				}
				else
				{
					t = v + dist;

					if (t > EPSILON) 
					{
						dist = t;
					}
					else
					{
						dist = 0;
					}
				}
			}
#endif

			isection.object_ = object;
			isection.ray_    = ray;
			isection.dist_   = dist;
		}
	}

	return isection;
}

cl_float3 normal(Object object, cl_float3 pos)
{
	switch(object.type_) 
	{
		case OBJECT_PLANE:
		{
			return(object.centerOrNorm_);
			break;
		}
		case OBJECT_SPHERE:
		{
			cl_float3 center = object.centerOrNorm_;
			return normalize(pos - center);
			break;
		}
	}
}

__kernel void initializeCamera(
	float3 position, 
	float3 lookAt,
	__global Camera * camera)
{
	camera->forward_ = normalize(lookAt - position);

	camera->right_   = 
		1.5f * normalize(
				cross(camera->forward_, (float3)(0.f, -1.f, 0.f)));
					 
	camera->up_   = 
		1.5f * normalize(
				cross(camera->forward_, camera->right_));

	camera->position_ = position;
#if 0
	camera->forward_ = (float3)(-0.6384138f, -0.2858569f, 0.71464231f);
	camera->right_ = (float3)(-1.11864181f, 0.0f, 0.999320f);
	camera->up_    = (float3)(-0.2856625f, 1.437408f, -0.319775f);


	camera->position_.x = 2.75f;
	camera->position_.y = 2.0f;
	camera->position_.z = 3.75f;	
#endif
}

__kernel void initializeLight(
	float3 position,
	float3 colour,
	__global Light * lights,
	uint num)
{
	lights[num].position_ = position;
	lights[num].colour_   = colour;

	//PRINT(printf("colour(%f,%f,%f)\n", lights[num].position_.x, lights[num].position_.y, lights[num].position_.z));
	//PRINT(printf("colour(%f,%f,%f)\n", lights[num].colour_.x, lights[num].colour_.y, lights[num].colour_.z));
}

#if 0
__kernel void initializeObject(
	int type,
	float3 centerOrNorm,
	float radiusOrOffset,
	Surface surface,
	__global Object * objects,
	uint num)
{	
	objects[num].type_           = type;

	objects[num].centerOrNorm_   = centerOrNorm;
	objects[num].radiusOrOffset_ = radiusOrOffset;
	objects[num].surface_        = surface;
}

#endif

__kernel void initializeObject(
	__global Object * objects)
{	
#if 0
	objects[0].type_               = OBJECT_SPHERE;
	objects[0].centerOrNorm_       = (float3)(-1.65f, 0.5f, 2.5f);
	objects[0].radiusOrOffset_     = 0.75f;
	objects[0].surface_.type_      = SURFACE_MATT_SHINY;
	objects[0].surface_.roughness_ = 250.0f;

	objects[1].type_               = OBJECT_SPHERE;
	objects[1].centerOrNorm_       = (float3)(2.65f, -4.0f, -0.25f);
	objects[1].radiusOrOffset_     = 2.2f;
	objects[1].surface_.type_      = SURFACE_SHINY;
	objects[1].surface_.roughness_ = 250.0f;

	objects[2].type_               = OBJECT_PLANE;
	objects[2].centerOrNorm_       = (float3)(0.0f, 1.0f, 0.0f);
	objects[2].radiusOrOffset_     = 0.0f;
	objects[2].surface_.type_      = SURFACE_CHECKERBOARD;
	objects[2].surface_.roughness_ = 150.0f;
#else
	objects[0].type_               = OBJECT_SPHERE;
	objects[0].centerOrNorm_       = (float3)(-0.5f, 1.0f, 1.5f);
	printf("x = %f\n", objects[0].centerOrNorm_.x);
	objects[0].radiusOrOffset_     = 0.75f;
	objects[0].surface_.type_      = SURFACE_MATT_SHINY;
	objects[0].surface_.roughness_ = 250.0f;

	objects[1].type_               = OBJECT_SPHERE;
	objects[1].centerOrNorm_       = (float3)(0.0f, 1.0f, -0.25f);
	objects[1].radiusOrOffset_     = 1.0f;
	objects[1].surface_.type_      = SURFACE_SHINY;
	objects[1].surface_.roughness_ = 250.0f;

	objects[2].type_               = OBJECT_PLANE;
	objects[2].centerOrNorm_       = (float3)(0.0f, 1.0f, 0.0f);
	objects[2].radiusOrOffset_     = 0.0f;
	objects[2].surface_.type_      = SURFACE_CHECKERBOARD;
	objects[2].surface_.roughness_ = 150.0f;

#endif
}

//----------------------------------------------------------------------------------------

float recenterX(float x)
{
	float size = convert_float(get_global_size(0));
	return (x - (size / 2.0f)) / (2.0f * size);
}

float recenterY(float y)
{
	float size = convert_float(get_global_size(1));	
	return -(y - (size / 2.0f)) / (2.0f * size);
}

float3 getPoint(float x, float y, Camera camera)
{
	return normalize(
		camera.forward_ + 
		(recenterX(x) * camera.right_ + 
		 recenterY(y) * camera.up_));
}

InterSection minInterSection(
	Ray ray,
	__global Object * objects,
	uint numObjects)
{
	InterSection min;
	min.dist_ = 0;

	for (uint i = 0; i < numObjects; i++) 
	{
		InterSection isection = intersect(objects[i], ray);
		if (isection.dist_ != 0)
		{
			if (min.dist_ == 0 || min.dist_ > isection.dist_)
			{
				min = isection;
			}
		}
	}

	return min;
}

float testRay(
	Ray ray, 
	__global Object * objects,
	uint numObjects)
{
	InterSection isection = minInterSection(ray, objects, numObjects);
	if (isection.dist_ == 0)
	{
		return 0;
	}

	return isection.dist_;
}

float3 getNaturalColour(
	Object object,
	float3 position,
	float3 norm,
	float3 rd,
	__global Object * objects,
	uint numObjects,
	__global Light * lights,
	uint numLights)
{
	float3 colour = (float3)(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < numLights; i++)
	{
		float3 ldis  = lights[i].position_ - position;
		float3 livec = normalize(ldis);

		Ray ray;
		ray.start_     = position;
		ray.direction_ = livec;

		float neatISection = testRay(ray, objects, numObjects);
		bool inShadow = !((neatISection > length(ldis) - EPSILON) || (neatISection == 0));
		if (!inShadow)
		{
			float illum = dot(livec, norm);
			float3 lcolour;
			if (illum > 0.0f)
			{
				lcolour = illum * lights[i].colour_;
			}
			else
			{
				lcolour = DEFAULT_COLOUR;
			}
			float specular = dot(livec, normalize(rd));
			float3 scolour;
			if (specular > 0.0f)
			{
				scolour = pow(
					specular, 
					convert_int(object.surface_.roughness_)) * lights[i].colour_; 
			}
			else
			{
				scolour = (float3)(1.0f, 1.0f, 0.0f);
			}

			colour = 
				colour + 
					(surfaceDiffuse(position, object.surface_) * lcolour + 
					 surfaceSpecular(position, object.surface_) * scolour);
		}
	}

	return colour;
}

float3 shade(
	__local InterSection * isection,
	__global Object * objects,
	uint numObjects,
	__global Light * lights,
	uint numLights, 
	__local int * depth,
	__local float3 * norm,
	__local float3 * reflectDirection,
	__local float3 * position,
	int offset)
{
	float3 d        = (isection+offset)->ray_.direction_;
	*(position+offset)       = (isection+offset)->dist_ * (isection+offset)->ray_.direction_ + (isection+offset)->ray_.start_;
	*(norm+offset)           = normal((isection+offset)->object_, *(position+offset));
	*(reflectDirection+offset) = d - (2 * dot(*(norm+offset),d)) * *(norm+offset);
	float3 colour   = DEFAULT_COLOUR;
	
	colour = colour + getNaturalColour(
		(isection+offset)->object_,
		*(position+offset),
		*(norm+offset),
		*(reflectDirection+offset),
		objects,
		numObjects,
		lights,
		numLights);

	// Handle the exit case, i.e. recusion depth reached 
	if (*(depth+offset) >= MAX_DEPTH-1)
	{
		colour = colour + (float3)(0.5f, 0.5f, 0.5f);
	}

	return colour;
}

float3 traceRay(
	Ray ray, 
	__global Object * objects,
	uint numObjects,
	__global Light * lights,
	uint numLights, 
	__local int * depth,
	__local InterSection * isection,
	__local float3 * normal,
	__local float3 * direction,
	__local float3 * position,
	int offset)
{
	*(isection+offset) = minInterSection(ray, objects, numObjects);
	if ((isection+offset)->dist_ == 0)
	{
		*(depth+offset) = MAX_DEPTH+1; // early exit
		return BACKGROUND_COLOUR;
	}

	return shade(
		isection, 
		objects, 
		numObjects, 
		lights, 
		numLights, 
		depth,
		normal,
		direction,
		position,
		offset);
}

__kernel void render(
	__global Camera * camera,
	__global Object * objects,
	uint numObjects, 
	__global uchar4 * output,
	uint stride,
	__global Light * lights,
	uint numLights,
	__local InterSection * isection,
	__local float3 * normal,
	__local float3 * direction,
	__local float3 * position,
	__local int * depth)
{
	uint tmp = 0;
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);

	Ray ray;
	ray.start_     = camera->position_;
	ray.direction_ = getPoint(x, y, *camera);

	// Pre-calulate the local memory addresses for each work-item
	// we need these to handle pass results back for the next iteration
	size_t offset = get_local_id(0)*get_local_size(0)+get_local_id(1);
#if 0
	isection  = isection + offset;
	normal    = normal + offset;
	direction = direction + offset;
	position  = position + offset;
	depth     = depth + offset;
#endif
	*(depth+offset)    = 0;

	// Trace ray from camera
	float3 colour = traceRay(
		ray, 
		objects, 
		numObjects, 
		lights, 
		numLights, 
		depth,
		isection,
		normal,
		direction,
		position,
		offset);

	*(depth+offset) = *depth + 1;
	
	// Trace reflection rays
	while(*(depth+offset) < MAX_DEPTH)
	{
		Ray refRay;
		ray.start_     = *(position+offset) + 0.001f * (*(direction+offset));
		ray.direction_ = *(direction+offset);

		float surRef = surfaceReflect(*(position+offset), (isection+offset)->object_.surface_);

		float3 refColour = 
			surRef *
			traceRay(
				refRay, 
				objects, 
				numObjects, 
				lights, 
				numLights, 
				depth,
				isection,
				normal,
				direction,
				position,
				offset);

		colour  = colour +  refColour;
			
		*(depth+offset) = *(depth+offset) + 1;
	}

	uchar4 colouru = (uchar4)(
		toUChar(colour.x),
		toUChar(colour.y), 
		toUChar(colour.z), 0xFF);

	output[x*stride+y] = colouru;
}